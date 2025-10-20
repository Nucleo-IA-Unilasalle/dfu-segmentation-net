import os
from typing import Tuple
import kagglehub
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import mlflow
import time

from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

# Download latest version
path = kagglehub.dataset_download("leoscode/wound-segmentation-images")

train_images = path + "/data_wound_seg/train_images"
train_masks = path + "/data_wound_seg/train_masks"

test_images = path + "/data_wound_seg/test_images"
test_masks = path + "/data_wound_seg/test_masks"


class WoundSegmentationDataset(Dataset):
    """Dataset class for loading wound images and binary masks."""
    
    def __init__(self, image_dir: str, mask_dir: str, transform: transforms.Compose = None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        # Load image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        # Binarize mask
        mask = (mask > 0.5).float()
        
        return image, mask


class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation."""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class EarlyStopping:
    """Early stopping utility to stop training when validation metric stops improving."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min', restore_best_weights: bool = True):
        """
        Args:
            patience: Number of epochs to wait after last improvement
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' for loss (lower is better), 'max' for accuracy/IoU/Dice (higher is better)
            restore_best_weights: Whether to restore model weights from the best epoch
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None
        
    def __call__(self, val_score: float, model: nn.Module) -> bool:
        """
        Check if training should stop early.
        
        Args:
            val_score: Current validation score
            model: Model to potentially save weights from
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = val_score
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        elif self._is_better(val_score, self.best_score):
            self.best_score = val_score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    print(f"Restored best weights with score: {self.best_score:.4f}")
        
        return self.early_stop
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current score is better than best score."""
        if self.mode == 'min':
            return current < best - self.min_delta
        else:  # mode == 'max'
            return current > best + self.min_delta


class UNetBlock(nn.Module):
    """Basic convolutional block for U-Net."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class UNet(nn.Module):
    """Simple U-Net architecture for binary segmentation."""
    
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        
        # Encoder
        self.enc1 = UNetBlock(in_channels, 64)
        self.enc2 = UNetBlock(64, 128)
        self.enc3 = UNetBlock(128, 256)
        self.enc4 = UNetBlock(256, 512)
        
        # Bottleneck
        self.bottleneck = UNetBlock(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = UNetBlock(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = UNetBlock(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = UNetBlock(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = UNetBlock(128, 64)
        
        # Output
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        return torch.sigmoid(self.out(dec1))


class EfficientNetUNet(nn.Module):
    """U-Net architecture with EfficientNet-B4 encoder for binary segmentation."""
    
    def __init__(self, out_channels: int = 1, pretrained: bool = True):
        super().__init__()
        
        # Load pretrained EfficientNet-B4 as encoder
        if pretrained:
            weights = EfficientNet_B4_Weights.IMAGENET1K_V1
            efficientnet = efficientnet_b4(weights=weights)
        else:
            efficientnet = efficientnet_b4(weights=None)
        
        # Extract encoder features at different stages
        # EfficientNet-B4 channel outputs (actual):
        # Stage 0: 48 channels
        # Stage 1: 24 channels
        # Stage 2: 32 channels
        # Stage 3: 56 channels
        # Stage 4: 112 channels
        # Stage 5: 160 channels
        # Stage 6: 272 channels
        # Stage 7: 448 channels
        # Stage 8: 1792 channels
        self.encoder = efficientnet.features
        
        # Decoder with skip connections
        # Spatial dimensions: 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128 -> 256x256
        
        # Bottleneck to dec5: 1792 -> 272 (8x8, no upsampling, skip with stage 6)
        self.dec5 = UNetBlock(1792 + 272, 272)  # 1792 + 272 from stage 6
        
        # dec5 to dec4: 272 -> 160 (8x8 -> 16x16, skip with stage 5)
        self.upconv4 = nn.ConvTranspose2d(272, 160, kernel_size=2, stride=2)
        self.dec4 = UNetBlock(160 + 160, 160)  # 160 + 160 from stage 5
        
        # dec4 to dec3: 160 -> 56 (16x16 -> 32x32, skip with stage 3)
        self.upconv3 = nn.ConvTranspose2d(160, 56, kernel_size=2, stride=2)
        self.dec3 = UNetBlock(56 + 56, 56)  # 56 + 56 from stage 3
        
        # dec3 to dec2: 56 -> 32 (32x32 -> 64x64, skip with stage 2)
        self.upconv2 = nn.ConvTranspose2d(56, 32, kernel_size=2, stride=2)
        self.dec2 = UNetBlock(32 + 32, 32)  # 32 + 32 from stage 2
        
        # dec2 to dec1: 32 -> 24 (64x64 -> 128x128, skip with stage 1)
        self.upconv1 = nn.ConvTranspose2d(32, 24, kernel_size=2, stride=2)
        self.dec1 = UNetBlock(24 + 24, 24)  # 24 + 24 from stage 1
        
        # Final upsampling to original resolution (128x128 -> 256x256)
        self.upconv0 = nn.ConvTranspose2d(24, 16, kernel_size=2, stride=2)
        self.dec0 = UNetBlock(16, 16)
        
        # Output
        self.out = nn.Conv2d(16, out_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder with skip connections
        # Extract features at different stages for skip connections
        
        # Stage 0: Initial conv (48 channels, 128x128)
        enc0 = self.encoder[0](x)
        
        # Stage 1: (24 channels, 128x128)
        enc1 = self.encoder[1](enc0)
        
        # Stage 2: (32 channels, 64x64)
        enc2 = self.encoder[2](enc1)
        
        # Stage 3: (56 channels, 32x32)
        enc3 = self.encoder[3](enc2)
        
        # Stage 4: (112 channels, 16x16) - not using this for skip
        enc4 = self.encoder[4](enc3)
        
        # Stage 5: (160 channels, 16x16)
        enc5 = self.encoder[5](enc4)
        
        # Stage 6: (272 channels, 8x8)
        enc6 = self.encoder[6](enc5)
        
        # Stage 7: (448 channels, 8x8) - not using this for skip
        enc7 = self.encoder[7](enc6)
        
        # Stage 8: Bottleneck (1792 channels, 8x8)
        bottleneck = self.encoder[8](enc7)
        
        # Decoder with skip connections
        # dec5: Concatenate with enc6 (same spatial size 8x8)
        dec5 = torch.cat([bottleneck, enc6], dim=1)  # 1792 + 272
        dec5 = self.dec5(dec5)
        
        # dec4: Upsample to 16x16, concatenate with enc5
        dec4 = self.upconv4(dec5)
        dec4 = torch.cat([dec4, enc5], dim=1)  # 160 + 160
        dec4 = self.dec4(dec4)
        
        # dec3: Upsample to 32x32, concatenate with enc3
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)  # 56 + 56
        dec3 = self.dec3(dec3)
        
        # dec2: Upsample to 64x64, concatenate with enc2
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)  # 32 + 32
        dec2 = self.dec2(dec2)
        
        # dec1: Upsample to 128x128, concatenate with enc1
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)  # 24 + 24
        dec1 = self.dec1(dec1)
        
        # dec0: Upsample to 256x256
        dec0 = self.upconv0(dec1)
        dec0 = self.dec0(dec0)
        
        return torch.sigmoid(self.out(dec0))


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Tuple[float, float, float]:
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    batch_idx = 0
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate IoU and Dice for this batch
        batch_iou = calculate_iou(outputs, masks)
        batch_dice = calculate_dice(outputs, masks)
        total_iou += batch_iou
        total_dice += batch_dice
        
        # Log batch metrics every 10 batches
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch}] Batch [{batch_idx}/{len(dataloader)}] - Loss: {loss.item():.4f}, IoU: {batch_iou:.4f}, Dice: {batch_dice:.4f}")
            
            # Log batch metrics to MLflow
            mlflow.log_metric("batch/loss", loss.item(), step=epoch * len(dataloader) + batch_idx)
            mlflow.log_metric("batch/iou", batch_iou, step=epoch * len(dataloader) + batch_idx)
            mlflow.log_metric("batch/dice", batch_dice, step=epoch * len(dataloader) + batch_idx)
        
        batch_idx += 1

    avg_loss = total_loss / len(dataloader)
    avg_iou = total_iou / len(dataloader)
    avg_dice = total_dice / len(dataloader)
    return avg_loss, avg_iou, avg_dice


def calculate_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Calculate Intersection over Union (IoU) for binary segmentation."""
    # Apply threshold to predictions
    pred_binary = (pred > threshold).float()
    
    # Calculate intersection and union
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection
    
    # Avoid division by zero
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return (intersection / union).item()


def calculate_dice(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> float:
    """Calculate Dice coefficient for binary segmentation."""
    # Apply threshold to predictions
    pred_binary = (pred > threshold).float()
    
    # Calculate intersection and sums
    intersection = (pred_binary * target).sum()
    pred_sum = pred_binary.sum()
    target_sum = target.sum()
    
    # Calculate Dice coefficient
    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
    
    return dice.item()


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float]:
    """Validate model for one epoch."""
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Calculate IoU and Dice for this batch
            batch_iou = calculate_iou(outputs, masks)
            batch_dice = calculate_dice(outputs, masks)
            
            total_loss += loss.item()
            total_iou += batch_iou
            total_dice += batch_dice
    
    avg_loss = total_loss / len(dataloader)
    avg_iou = total_iou / len(dataloader)
    avg_dice = total_dice / len(dataloader)
    
    return avg_loss, avg_iou, avg_dice


def main() -> None:
    """Main training function."""
    # Hyperparameters
    batch_size = 8
    learning_rate = 0.001
    num_epochs = 15
    image_size = 256
    
    # Early stopping parameters
    early_stopping_patience = 5
    early_stopping_min_delta = 0.001
    early_stopping_metric = 'dice'  # 'loss', 'iou', or 'dice'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize MLflow
    mlflow.set_experiment("wound_segmentation_unet")
    mlflow.start_run()
    
    # Log script content to MLflow
    with open(__file__, 'r', encoding='utf-8') as f:
        script_content = f.read()
    mlflow.log_text(script_content, "code.py")
    
    # Log hyperparameters
    mlflow.log_param("model", "EfficientNet-B4-UNet")
    mlflow.log_param("encoder", "EfficientNet-B4")
    mlflow.log_param("pretrained", "ImageNet")
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("image_size", image_size)
    mlflow.log_param("device", str(device))
    mlflow.log_param("early_stopping_patience", early_stopping_patience)
    mlflow.log_param("early_stopping_min_delta", early_stopping_min_delta)
    mlflow.log_param("early_stopping_metric", early_stopping_metric)
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    # Datasets
    train_dataset = WoundSegmentationDataset(train_images, train_masks, transform)
    test_dataset = WoundSegmentationDataset(test_images, test_masks, transform)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    # Log dataset info
    mlflow.log_param("train_samples", len(train_dataset))
    mlflow.log_param("test_samples", len(test_dataset))
    
    # Model, loss, optimizer
    model = EfficientNetUNet(out_channels=1, pretrained=True).to(device)
    criterion = DiceLoss(smooth=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Log dynamic hyperparameters
    mlflow.log_param("optimizer", optimizer.__class__.__name__)
    mlflow.log_param("loss_function", criterion.__class__.__name__)
    
    # Initialize early stopping
    early_stopping_mode = 'max' if early_stopping_metric in ['iou', 'dice'] else 'min'
    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta,
        mode=early_stopping_mode,
        restore_best_weights=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    best_val_iou = 0.0
    best_val_dice = 0.0
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        train_loss, train_iou, train_dice = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_iou, val_dice = validate_epoch(model, test_loader, criterion, device)
        
        epoch_duration = time.time() - epoch_start_time
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}, Train Dice: {train_dice:.4f}, Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}, Val Dice: {val_dice:.4f}, Duration: {epoch_duration:.2f}s")
        
        # Log grouped metrics to MLflow
        mlflow.log_metric("train/loss", train_loss, step=epoch)
        mlflow.log_metric("train/iou", train_iou, step=epoch)
        mlflow.log_metric("train/dice", train_dice, step=epoch)
        mlflow.log_metric("val/loss", val_loss, step=epoch)
        mlflow.log_metric("val/iou", val_iou, step=epoch)
        mlflow.log_metric("val/dice", val_dice, step=epoch)
        mlflow.log_metric("train/epoch_duration", epoch_duration, step=epoch)
        
        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_filename = os.path.splitext(os.path.basename(__file__))[0] + "_best_efficientnet_b4_unet_model.pth"
            torch.save(model.state_dict(), model_filename)
            print(f"Saved best model with val loss: {val_loss:.4f}")
            mlflow.log_metric("best/val_loss", best_val_loss, step=epoch)
        
        # Track best IoU
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            mlflow.log_metric("best/val_iou", best_val_iou, step=epoch)
        
        # Track best Dice
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            mlflow.log_metric("best/val_dice", best_val_dice, step=epoch)
        
        # Check early stopping
        if early_stopping_metric == 'loss':
            val_score = val_loss
        elif early_stopping_metric == 'iou':
            val_score = val_iou
        elif early_stopping_metric == 'dice':
            val_score = val_dice
        else:
            raise ValueError(f"Invalid early stopping metric: {early_stopping_metric}")
        
        if early_stopping(val_score, model):
            print(f"Early stopping triggered at epoch {epoch+1}")
            print(f"Best {early_stopping_metric}: {early_stopping.best_score:.4f}")
            mlflow.log_param("early_stopping_triggered", True)
            mlflow.log_param("early_stopping_epoch", epoch+1)
            mlflow.log_param("early_stopping_best_score", early_stopping.best_score)
            break
        else:
            mlflow.log_metric("early_stopping/counter", early_stopping.counter, step=epoch)
    
    # Log early stopping status if not triggered
    if not early_stopping.early_stop:
        mlflow.log_param("early_stopping_triggered", False)
        mlflow.log_param("early_stopping_best_score", early_stopping.best_score)
    
    # Log final best metrics
    mlflow.log_metric("final/best_val_loss", best_val_loss)
    mlflow.log_metric("final/best_val_iou", best_val_iou)
    mlflow.log_metric("final/best_val_dice", best_val_dice)
    
    # Log model artifact
    model_filename = os.path.splitext(os.path.basename(__file__))[0] + "_best_efficientnet_b4_unet_model.pth"
    mlflow.log_artifact(model_filename)
    
    # End MLflow run
    mlflow.end_run()
    
    print("Training completed!")


if __name__ == "__main__":
    main()
