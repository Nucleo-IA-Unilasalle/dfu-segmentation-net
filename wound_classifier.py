import os
import json
import time
import random
import shutil
from typing import Tuple, List, Dict, Any
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import kagglehub
from datasets import load_dataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import mlflow

from pretrained import EfficientNetUNet, WoundSegmentationDataset


class WoundVerificationDataset(Dataset):
    """Dataset for wound verification that combines images with predicted masks."""
    
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        segmentation_model: nn.Module,
        device: torch.device,
        transform: transforms.Compose = None,
        cache_dir: str = "mask_cache",
        use_cache: bool = True
    ):
        """
        Args:
            image_paths: List of paths to images
            labels: List of labels (1 for wound, 0 for non-wound)
            segmentation_model: Trained segmentation model for generating masks
            device: Device to run segmentation model on
            transform: Transform to apply to images
            cache_dir: Directory to cache generated masks
            use_cache: Whether to use cached masks
        """
        self.image_paths = image_paths
        self.labels = labels
        self.segmentation_model = segmentation_model
        self.device = device
        self.transform = transform
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Pre-generate all masks if not cached
        if self.use_cache:
            self._generate_all_masks()
    
    def _get_cache_path(self, image_path: str) -> str:
        """Get cache file path for a given image path."""
        # Create a unique cache filename from the image path
        cache_filename = str(hash(image_path)) + ".npy"
        return os.path.join(self.cache_dir, cache_filename)
    
    def _generate_mask(self, image_path: str) -> np.ndarray:
        """Generate predicted mask for an image."""
        # Load and transform image
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Generate prediction
        self.segmentation_model.eval()
        with torch.no_grad():
            mask_pred = self.segmentation_model(image_tensor)
        
        # Convert to numpy array
        mask_np = mask_pred.squeeze().cpu().numpy()
        return mask_np

    # BUG: This doesn't seem to be identifying already generated masks,
    # thus regenerating all the masks everytime we try to train
    def _generate_all_masks(self) -> None:
        """Pre-generate all masks and cache them."""
        print("Checking mask cache...")
        masks_to_generate = []
        
        for image_path in self.image_paths:
            cache_path = self._get_cache_path(image_path)
            if not os.path.exists(cache_path):
                masks_to_generate.append(image_path)
        
        if len(masks_to_generate) > 0:
            print(f"Generating {len(masks_to_generate)} masks...")
            for idx, image_path in enumerate(masks_to_generate):
                if idx % 100 == 0:
                    print(f"  Generated {idx}/{len(masks_to_generate)} masks")
                
                mask_np = self._generate_mask(image_path)
                cache_path = self._get_cache_path(image_path)
                np.save(cache_path, mask_np)
            print(f"All masks generated and cached!")
        else:
            print("All masks already cached!")
    
    def _load_mask(self, image_path: str) -> np.ndarray:
        """Load mask from cache or generate it."""
        if self.use_cache:
            cache_path = self._get_cache_path(image_path)
            if os.path.exists(cache_path):
                return np.load(cache_path)
        
        return self._generate_mask(image_path)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Returns:
            combined_input: 4-channel tensor (RGB + mask)
            segmentation_percentage: Scalar tensor with % of segmented pixels
            label: Binary label (1 for wound, 0 for non-wound)
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image_tensor = self.transform(image)
        
        # Load or generate mask
        mask_np = self._load_mask(image_path)
        mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0)
        
        # Calculate segmentation percentage
        total_pixels = mask_np.size
        segmented_pixels = (mask_np > 0.5).sum()
        seg_percentage = (segmented_pixels / total_pixels) * 100.0
        seg_percentage_tensor = torch.tensor([seg_percentage], dtype=torch.float32)
        
        # Combine image and mask (4 channels)
        combined_input = torch.cat([image_tensor, mask_tensor], dim=0)
        
        return combined_input, seg_percentage_tensor, label


class WoundVerificationModel(nn.Module):
    """CNN model for wound verification from image + mask + percentage."""
    
    def __init__(self, dropout_rate: float = 0.5):
        """
        Args:
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        
        # CNN backbone for 4-channel input (RGB + mask)
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # 256 -> 128
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 128 -> 64
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # 64 -> 32
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)  # 32 -> 16
        
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers with percentage feature fusion
        self.fc1 = nn.Linear(512 + 1, 256)  # 512 from CNN + 1 from percentage
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(128, 1)  # Binary classification
        
    def forward(self, x: torch.Tensor, seg_percentage: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 4-channel input (batch_size, 4, 256, 256)
            seg_percentage: Segmentation percentage (batch_size, 1)
            
        Returns:
            Logits for binary classification (batch_size, 1)
        """
        # CNN backbone
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        
        x = F.relu(self.bn5(self.conv5(x)))
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Concatenate with percentage feature
        x = torch.cat([x, seg_percentage], dim=1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x


def load_segmentation_model(model_path: str, device: torch.device) -> nn.Module:
    """Load the trained segmentation model."""
    model = EfficientNetUNet(out_channels=1, pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def get_wound_image_paths(split: str = "train") -> List[str]:
    """Get paths to wound images from multiple datasets."""
    image_paths = []
    
    # 1. Original Wound Segmentation Dataset
    try:
        path = kagglehub.dataset_download("leoscode/wound-segmentation-images")
        if split == "train":
            image_dir = os.path.join(path, "data_wound_seg", "train_images")
        else:
            image_dir = os.path.join(path, "data_wound_seg", "test_images")
        
        if os.path.exists(image_dir):
            for filename in os.listdir(image_dir):
                if filename.endswith('.png'):
                    image_paths.append(os.path.join(image_dir, filename))
    except Exception as e:
        print(f"Warning: Could not load original wound dataset: {e}")

    # 2. Leprosy Chronic Wound Dataset
    try:
        leprosy_path = kagglehub.dataset_download("orvile/leprosy-chronic-wound-images-co2wounds-v2")
        # Use the recursive helper to find all images
        leprosy_images = []
        extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
        for root, dirs, files in os.walk(leprosy_path):
            for file in files:
                if file.endswith(extensions):
                    leprosy_images.append(os.path.join(root, file))
        
        # Deterministic split based on filename sorting
        leprosy_images.sort()
        split_idx = int(0.8 * len(leprosy_images))
        
        if len(leprosy_images) > 0:
            if split == "train":
                added_images = leprosy_images[:split_idx]
                image_paths.extend(added_images)
                print(f"Added {len(added_images)} leprosy wound images to train split")
            else:
                added_images = leprosy_images[split_idx:]
                image_paths.extend(added_images)
                print(f"Added {len(added_images)} leprosy wound images to test split")
    except Exception as e:
        print(f"Warning: Could not load leprosy wound dataset: {e}")

    return sorted(image_paths)


def get_non_wound_image_paths(download_dir: str = "datasets") -> List[str]:
    """Get paths to non-wound images from FP evaluation datasets."""
    all_paths = []
    
    # MiniImageNet
    try:
        mini_path = kagglehub.dataset_download("deeptrial/miniimagenet")
        mini_images = get_image_paths_recursive(mini_path)
        all_paths.extend(mini_images)
        print(f"Found {len(mini_images)} MiniImageNet images")
    except Exception as e:
        print(f"Warning: Could not load MiniImageNet: {e}")
    
    # Skin Cancer
    try:
        skin_cancer_dir = os.path.join(download_dir, "skin_cancer")
        if os.path.exists(skin_cancer_dir):
            skin_cancer_images = get_image_paths_recursive(skin_cancer_dir)
            all_paths.extend(skin_cancer_images)
            print(f"Found {len(skin_cancer_images)} Skin Cancer images")
        else:
            # Download it
            dataset = load_dataset("Pranavkpba2000/skin_cancer_small_dataset", split="test")
            os.makedirs(skin_cancer_dir, exist_ok=True)
            for idx, item in enumerate(dataset):
                image_path = os.path.join(skin_cancer_dir, f"image_{idx}.jpg")
                if not os.path.exists(image_path):
                    item['image'].save(image_path)
                all_paths.append(image_path)
            print(f"Downloaded {len(dataset)} Skin Cancer images")
    except Exception as e:
        print(f"Warning: Could not load Skin Cancer dataset: {e}")
    
    # Skin Disease
    try:
        skin_disease_path = kagglehub.dataset_download("pacificrm/skindiseasedataset")
        skin_disease_images = get_image_paths_recursive(skin_disease_path)
        all_paths.extend(skin_disease_images)
        print(f"Found {len(skin_disease_images)} Skin Disease images")
    except Exception as e:
        print(f"Warning: Could not load Skin Disease dataset: {e}")
        
    # HAM10000 (Hard Negatives - Skin Lesions)
    try:
        ham_path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
        ham_images = get_image_paths_recursive(ham_path)
        # Downsample if too large (it has ~10k images) to prevent it from dominating
        # But we want to learn hard negatives well, so keeping all is probably fine 
        # as long as we balance classes later (which we do).
        all_paths.extend(ham_images)
        print(f"Found {len(ham_images)} HAM10000 skin lesion images")
    except Exception as e:
        print(f"Warning: Could not load HAM10000: {e}")
    
    return all_paths


def get_image_paths_recursive(directory: str) -> List[str]:
    """Recursively get all image paths from a directory."""
    extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths


def create_balanced_dataset(
    wound_paths: List[str],
    non_wound_paths: List[str],
    balance_size: int = None
) -> Tuple[List[str], List[int]]:
    """
    Create a balanced dataset with equal wound and non-wound samples.
    
    Args:
        wound_paths: List of wound image paths
        non_wound_paths: List of non-wound image paths
        balance_size: Number of samples per class (if None, use minimum)
        
    Returns:
        Tuple of (image_paths, labels)
    """
    if balance_size is None:
        balance_size = min(len(wound_paths), len(non_wound_paths))
    
    # Sample from each class
    wound_sample = random.sample(wound_paths, min(balance_size, len(wound_paths)))
    non_wound_sample = random.sample(non_wound_paths, min(balance_size, len(non_wound_paths)))
    
    # Combine
    image_paths = wound_sample + non_wound_sample
    labels = [1] * len(wound_sample) + [0] * len(non_wound_sample)
    
    # Shuffle together
    combined = list(zip(image_paths, labels))
    random.shuffle(combined)
    image_paths, labels = zip(*combined)
    
    return list(image_paths), list(labels)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_batches_per_epoch: int = None
) -> Tuple[float, float]:
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    if total_batches_per_epoch is None:
        total_batches_per_epoch = len(dataloader)
    
    # Calculate logging interval (every 5% of epoch)
    log_interval = max(1, total_batches_per_epoch // 20)
    
    for batch_idx, (images, percentages, labels) in enumerate(dataloader):
        images = images.to(device)
        percentages = percentages.to(device)
        labels = labels.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(images, percentages)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        # Calculate step for MLflow (epoch * batches_per_epoch + batch_idx)
        step = epoch * total_batches_per_epoch + batch_idx
        
        # Log every 5% of epoch
        if batch_idx % log_interval == 0 or batch_idx == len(dataloader) - 1:
            running_loss = total_loss / (batch_idx + 1)
            running_acc = 100.0 * correct / total
            batch_loss = loss.item()
            mlflow.log_metrics({
                "train_loss": running_loss,
                "train_batch_loss": batch_loss,
                "train_accuracy": running_acc
            }, step=step)
        
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch}] Batch [{batch_idx}/{len(dataloader)}] - Loss: {loss.item():.4f}, Acc: {100.0 * correct / total:.2f}%")
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float, float, float, np.ndarray]:
    """
    Validate model for one epoch.
    
    Returns:
        Tuple of (loss, accuracy, precision, recall, f1, confusion_matrix)
    """
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, percentages, labels in dataloader:
            images = images.to(device)
            percentages = percentages.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            
            outputs = model(images, percentages)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            # Store predictions and labels
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions).flatten()
    all_labels = np.array(all_labels).flatten()
    
    # Calculate metrics
    accuracy = 100.0 * (all_predictions == all_labels).sum() / len(all_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='binary', zero_division=0
    )
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    
    return avg_loss, accuracy, precision, recall, f1, conf_matrix


def calculate_detailed_metrics(conf_matrix: np.ndarray) -> Dict[str, float]:
    """Calculate detailed metrics from confusion matrix."""
    tn = int(conf_matrix[0, 0])
    fp = int(conf_matrix[0, 1])
    fn = int(conf_matrix[1, 0])
    tp = int(conf_matrix[1, 1])
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    return {
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'specificity': float(specificity),
        'f1_score': float(f1_score),
        'false_positive_rate': float(false_positive_rate),
        'false_negative_rate': float(false_negative_rate)
    }


def plot_confusion_matrix(conf_matrix: np.ndarray, output_path: str) -> None:
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Non-Wound', 'Wound'],
        yticklabels=['Non-Wound', 'Wound']
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Wound Verification Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")


def main() -> None:
    """Main training function."""
    # Configuration
    segmentation_model_path = "pretrained_best_efficientnet_b4_unet_model.pth"
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 25
    image_size = 256
    dropout_rate = 0.5
    balance_size = None  # Use minimum available
    
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # MLflow setup
    mlflow.set_experiment("Wound_Classifier_CNN")
    mlflow.start_run()
    mlflow.log_params({
        "model_type": "CNN_EfficientNet_Backbone",
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "image_size": image_size,
        "dropout_rate": dropout_rate,
        "balance_size": str(balance_size),
        "segmentation_model": segmentation_model_path
    })
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check if segmentation model exists
    if not os.path.exists(segmentation_model_path):
        raise FileNotFoundError(f"Segmentation model not found: {segmentation_model_path}")
    
    # Load segmentation model
    print("Loading segmentation model...")
    segmentation_model = load_segmentation_model(segmentation_model_path, device)
    print("Segmentation model loaded!")
    
    # Get image paths
    print("\nGathering image paths...")
    wound_train_paths = get_wound_image_paths("train")
    wound_test_paths = get_wound_image_paths("test")
    non_wound_paths = get_non_wound_image_paths()
    
    print(f"Wound training images: {len(wound_train_paths)}")
    print(f"Wound test images: {len(wound_test_paths)}")
    print(f"Non-wound images: {len(non_wound_paths)}")
    
    # Split non-wound images into train/test (80/20)
    random.shuffle(non_wound_paths)
    split_idx = int(0.8 * len(non_wound_paths))
    non_wound_train_paths = non_wound_paths[:split_idx]
    non_wound_test_paths = non_wound_paths[split_idx:]
    
    print(f"Non-wound training images: {len(non_wound_train_paths)}")
    print(f"Non-wound test images: {len(non_wound_test_paths)}")
    
    # Create balanced datasets
    print("\nCreating balanced datasets...")
    train_paths, train_labels = create_balanced_dataset(
        wound_train_paths, non_wound_train_paths, balance_size
    )
    test_paths, test_labels = create_balanced_dataset(
        wound_test_paths, non_wound_test_paths, balance_size
    )
    
    print(f"Training samples: {len(train_paths)} ({sum(train_labels)} wound, {len(train_labels) - sum(train_labels)} non-wound)")
    print(f"Test samples: {len(test_paths)} ({sum(test_labels)} wound, {len(test_labels) - sum(test_labels)} non-wound)")
    
    
    # Create transforms with augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
    ])
    
    # Standard transform for validation
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    # Create datasets
    print("\nCreating datasets (this will generate and cache masks)...")
    train_dataset = WoundVerificationDataset(
        train_paths, train_labels, segmentation_model, device, train_transform, cache_dir="mask_cache_train"
    )
    test_dataset = WoundVerificationDataset(
        test_paths, test_labels, segmentation_model, device, test_transform, cache_dir="mask_cache_test"
    )
    
    # Create dataloaders
    # Possible improvements for trainment
    # add prefetch_factor
    num_workers = os.cpu_count() if os.cpu_count() is not None else 2
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers)
    
    # Create verification model
    print("\nCreating wound verification model...")
    model = WoundVerificationModel(dropout_rate=dropout_rate).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print("\nStarting training...\n")
    best_f1 = 0.0
    training_history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': []
    }
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, len(train_loader)
        )
        
        # Validate
        val_loss, val_acc, val_precision, val_recall, val_f1, conf_matrix = validate_epoch(
            model, test_loader, criterion, device
        )
        
        epoch_time = time.time() - epoch_start
        
        # Store metrics
        training_history['train_loss'].append(train_loss)
        training_history['train_accuracy'].append(train_acc)
        training_history['val_loss'].append(val_loss)
        training_history['val_accuracy'].append(val_acc)
        training_history['val_precision'].append(float(val_precision))
        training_history['val_recall'].append(float(val_recall))
        training_history['val_f1'].append(float(val_f1))
        
        print(f"\nEpoch [{epoch+1}/{num_epochs}] - Time: {epoch_time:.2f}s")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")
        print(f"  Confusion Matrix:\n{conf_matrix}")

        # Log validation metrics to MLflow at end of epoch
        # Training metrics are already logged during epoch (every 5%)
        epoch_end_step = (epoch + 1) * len(train_loader)
        mlflow.log_metrics({
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_precision": float(val_precision),
            "val_recall": float(val_recall),
            "val_f1": float(val_f1)
        }, step=epoch_end_step)
        
        # Save best model based on F1 score
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "wound_classifier_best_model.pth")
            print(f"  Saved best model with F1: {val_f1:.4f}")
    
    # Plot final confusion matrix
    print("\nTraining completed!")
    print(f"Best validation F1 score: {best_f1:.4f}")
    
    # Load best model and evaluate
    model.load_state_dict(torch.load("wound_classifier_best_model.pth"))
    _, final_acc, final_precision, final_recall, final_f1, final_conf_matrix = validate_epoch(
        model, test_loader, criterion, device
    )
    
    plot_confusion_matrix(final_conf_matrix, "wound_classifier_confusion_matrix.png")
    
    # Calculate detailed metrics
    detailed_metrics = calculate_detailed_metrics(final_conf_matrix)
    
    # Save metrics
    final_metrics = {
        'best_f1_score': float(best_f1),
        'final_accuracy': float(final_acc),
        'final_precision': float(final_precision),
        'final_recall': float(final_recall),
        'final_f1': float(final_f1),
        'confusion_matrix': final_conf_matrix.tolist(),
        'detailed_metrics': detailed_metrics,
        'training_history': training_history,
        'hyperparameters': {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'image_size': image_size,
            'dropout_rate': dropout_rate
        }
    }
    
    with open('wound_classifier_metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    print(f"\nFinal metrics saved to: wound_classifier_metrics.json")
    print(f"\nFinal Test Results:")
    print(f"  Accuracy: {final_acc:.2f}%")
    print(f"  Precision: {final_precision:.4f}")
    print(f"  Recall: {final_recall:.4f}")
    print(f"  F1 Score: {final_f1:.4f}")
    print(f"  Specificity: {detailed_metrics['specificity']:.4f}")
    print(f"  False Positive Rate: {detailed_metrics['false_positive_rate']:.4f}")
    print(f"  False Negative Rate: {detailed_metrics['false_negative_rate']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN: {final_conf_matrix[0, 0]}, FP: {final_conf_matrix[0, 1]}")
    print(f"  FN: {final_conf_matrix[1, 0]}, TP: {final_conf_matrix[1, 1]}")

    # Log final metrics and artifacts to MLflow
    mlflow.log_metrics({
        "test_accuracy": final_acc,
        "test_precision": final_precision,
        "test_recall": final_recall,
        "test_f1": final_f1
    })
    
    mlflow.log_artifact("wound_classifier_metrics.json")
    if os.path.exists("wound_classifier_confusion_matrix.png"):
        mlflow.log_artifact("wound_classifier_confusion_matrix.png")
    if os.path.exists("wound_classifier_best_model.pth"):
        mlflow.log_artifact("wound_classifier_best_model.pth")
        
    mlflow.end_run()


if __name__ == "__main__":
    main()

