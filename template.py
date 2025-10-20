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


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Tuple[float, float]:
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0
    total_iou = 0.0
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
        
        # Calculate IoU for this batch
        batch_iou = calculate_iou(outputs, masks)
        total_iou += batch_iou
        
        # Log batch metrics every 10 batches
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch}] Batch [{batch_idx}/{len(dataloader)}] - Loss: {loss.item():.4f}, IoU: {batch_iou:.4f}")
            
            # Log batch metrics to MLflow
            mlflow.log_metric("batch/loss", loss.item(), step=epoch * len(dataloader) + batch_idx)
            mlflow.log_metric("batch/iou", batch_iou, step=epoch * len(dataloader) + batch_idx)
        
        batch_idx += 1

    avg_loss = total_loss / len(dataloader)
    avg_iou = total_iou / len(dataloader)
    return avg_loss, avg_iou


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


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Validate model for one epoch."""
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Calculate IoU for this batch
            batch_iou = calculate_iou(outputs, masks)
            
            total_loss += loss.item()
            total_iou += batch_iou
    
    avg_loss = total_loss / len(dataloader)
    avg_iou = total_iou / len(dataloader)
    
    return avg_loss, avg_iou


def main() -> None:
    """Main training function."""
    # Hyperparameters
    batch_size = 8
    learning_rate = 0.001
    num_epochs = 75
    image_size = 256
    
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
    mlflow.log_param("model", "UNet")
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("image_size", image_size)
    mlflow.log_param("device", str(device))
    
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
    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = DiceLoss(smooth=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Log dynamic hyperparameters
    mlflow.log_param("optimizer", optimizer.__class__.__name__)
    mlflow.log_param("loss_function", criterion.__class__.__name__)
    
    # Training loop
    best_val_loss = float('inf')
    best_val_iou = 0.0
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        train_loss, train_iou = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_iou = validate_epoch(model, test_loader, criterion, device)
        
        epoch_duration = time.time() - epoch_start_time
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}, Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}, Duration: {epoch_duration:.2f}s")
        
        # Log grouped metrics to MLflow
        mlflow.log_metric("train/loss", train_loss, step=epoch)
        mlflow.log_metric("train/iou", train_iou, step=epoch)
        mlflow.log_metric("val/loss", val_loss, step=epoch)
        mlflow.log_metric("val/iou", val_iou, step=epoch)
        mlflow.log_metric("train/epoch_duration", epoch_duration, step=epoch)
        
        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_filename = os.path.splitext(os.path.basename(__file__))[0] + "_best_unet_model.pth"
            torch.save(model.state_dict(), model_filename)
            print(f"Saved best model with val loss: {val_loss:.4f}")
            mlflow.log_metric("best/val_loss", best_val_loss, step=epoch)
        
        # Track best IoU
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            mlflow.log_metric("best/val_iou", best_val_iou, step=epoch)
    
    # Log final best metrics
    mlflow.log_metric("final/best_val_loss", best_val_loss)
    mlflow.log_metric("final/best_val_iou", best_val_iou)
    
    # Log model artifact
    model_filename = os.path.splitext(os.path.basename(__file__))[0] + "_best_unet_model.pth"
    mlflow.log_artifact(model_filename)
    
    # End MLflow run
    mlflow.end_run()
    
    print("Training completed!")


if __name__ == "__main__":
    main()
