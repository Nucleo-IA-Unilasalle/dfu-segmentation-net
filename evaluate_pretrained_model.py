import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import json
from typing import Tuple, List
import time
import kagglehub

# Import our model and utilities
from pretrained import EfficientNetUNet, WoundSegmentationDataset, calculate_iou, calculate_dice


def load_model(model_path: str, device: torch.device) -> nn.Module:
    """Load the trained model from checkpoint."""
    model = EfficientNetUNet(out_channels=1, pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5
) -> Tuple[float, float, float, List[float], List[float], List[float]]:
    """Evaluate model on test dataset."""
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    
    # Store individual scores for analysis
    iou_scores = []
    dice_scores = []
    loss_scores = []
    
    criterion = nn.BCELoss()  # Binary Cross Entropy for evaluation
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(dataloader):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, masks)
            
            # Calculate metrics for this batch
            batch_iou = calculate_iou(outputs, masks, threshold)
            batch_dice = calculate_dice(outputs, masks, threshold)
            
            # Accumulate metrics
            total_loss += loss.item()
            total_iou += batch_iou
            total_dice += batch_dice
            
            # Store individual scores
            iou_scores.append(batch_iou)
            dice_scores.append(batch_dice)
            loss_scores.append(loss.item())
            
            if batch_idx % 10 == 0:
                print(f"Batch [{batch_idx}/{len(dataloader)}] - Loss: {loss.item():.4f}, IoU: {batch_iou:.4f}, Dice: {batch_dice:.4f}")
    
    # Calculate averages
    avg_loss = total_loss / len(dataloader)
    avg_iou = total_iou / len(dataloader)
    avg_dice = total_dice / len(dataloader)
    
    return avg_loss, avg_iou, avg_dice, loss_scores, iou_scores, dice_scores


def evaluate_single_image(
    model: nn.Module,
    image_path: str,
    mask_path: str,
    device: torch.device,
    image_size: int = 256,
    threshold: float = 0.5
) -> Tuple[float, float, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate model on a single image."""
    model.eval()
    
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    
    # Apply transforms
    image_tensor = transform(image).unsqueeze(0).to(device)
    mask_tensor = transform(mask).unsqueeze(0).to(device)
    mask_tensor = (mask_tensor > 0.5).float()
    
    with torch.no_grad():
        pred_tensor = model(image_tensor)
        pred_binary = (pred_tensor > threshold).float()
    
    # Calculate metrics
    iou = calculate_iou(pred_tensor, mask_tensor, threshold)
    dice = calculate_dice(pred_tensor, mask_tensor, threshold)
    
    return iou, dice, image_tensor.cpu(), mask_tensor.cpu(), pred_tensor.cpu()




def print_statistics(scores: List[float], metric_name: str) -> None:
    """Print statistics for a list of scores."""
    scores_array = np.array(scores)
    print(f"\n{metric_name} Statistics:")
    print(f"  Mean: {scores_array.mean():.4f}")
    print(f"  Std:  {scores_array.std():.4f}")
    print(f"  Min:  {scores_array.min():.4f}")
    print(f"  Max:  {scores_array.max():.4f}")
    print(f"  Median: {np.median(scores_array):.4f}")


def evaluate_single_image_cli(model_path: str, image_path: str, mask_path: str, threshold: float = 0.5) -> None:
    """Evaluate model on a single image (CLI function)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = load_model(model_path, device)
    
    # Evaluate single image
    iou, dice, image_tensor, mask_tensor, pred_tensor = evaluate_single_image(
        model, image_path, mask_path, device, threshold=threshold
    )
    
    print(f"\nSingle Image Evaluation Results:")
    print(f"Image: {image_path}")
    print(f"Mask: {mask_path}")
    print(f"IoU: {iou:.4f}")
    print(f"Dice: {dice:.4f}")
    
    # Save results to JSON
    results = {
        'image_path': image_path,
        'mask_path': mask_path,
        'iou': float(iou),
        'dice': float(dice),
        'threshold': threshold
    }
    
    with open('single_image_evaluation.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: single_image_evaluation.json")


def main() -> None:
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate EfficientNet-B4 U-Net model")
    parser.add_argument("--model", default="pretrained_best_efficientnet_b4_unet_model.pth", 
                       help="Path to model checkpoint")
    parser.add_argument("--image", help="Path to single image for evaluation")
    parser.add_argument("--mask", help="Path to corresponding mask for single image evaluation")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary prediction")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    
    args = parser.parse_args()
    
    # Configuration
    model_path = args.model
    image_size = 256
    batch_size = args.batch_size
    threshold = args.threshold
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Please train the model first or check the model path.")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = load_model(model_path, device)
    print(f"Model loaded successfully!")
    
    # Single image evaluation
    if args.image and args.mask:
        if not os.path.exists(args.image) or not os.path.exists(args.mask):
            print("Image or mask file not found.")
            return
        evaluate_single_image_cli(model_path, args.image, args.mask, threshold)
        return
    
    # Full dataset evaluation
    print("Preparing test data...")
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    # Download dataset using kagglehub (same as training script)
    path = kagglehub.dataset_download("leoscode/wound-segmentation-images")
    
    test_images = path + "/data_wound_seg/test_images"
    test_masks = path + "/data_wound_seg/test_masks"
    
    # Check if test data exists
    if not os.path.exists(test_images) or not os.path.exists(test_masks):
        print("Test data paths not found. Please check the dataset download.")
        print(f"Expected test_images: {test_images}")
        print(f"Expected test_masks: {test_masks}")
        return
    
    test_dataset = WoundSegmentationDataset(test_images, test_masks, transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Evaluate model
    print("\nEvaluating model...")
    start_time = time.time()
    
    avg_loss, avg_iou, avg_dice, loss_scores, iou_scores, dice_scores = evaluate_model(
        model, test_loader, device, threshold
    )
    
    evaluation_time = time.time() - start_time
    
    # Print results
    print(f"\n{'='*50}")
    print("EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Evaluation time: {evaluation_time:.2f} seconds")
    print(f"Number of samples: {len(test_dataset)}")
    print(f"Threshold: {threshold}")
    print(f"\nAverage Metrics:")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  IoU:  {avg_iou:.4f}")
    print(f"  Dice: {avg_dice:.4f}")
    
    # Print detailed statistics
    print_statistics(loss_scores, "Loss")
    print_statistics(iou_scores, "IoU")
    print_statistics(dice_scores, "Dice")
    
    # Save detailed results to JSON
    results = {
        'avg_loss': float(avg_loss),
        'avg_iou': float(avg_iou),
        'avg_dice': float(avg_dice),
        'loss_scores': [float(score) for score in loss_scores],
        'iou_scores': [float(score) for score in iou_scores],
        'dice_scores': [float(score) for score in dice_scores],
        'threshold': threshold,
        'num_samples': len(test_dataset),
        'evaluation_time': evaluation_time
    }
    
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: evaluation_results.json")
    
    print(f"\nEvaluation completed!")


if __name__ == "__main__":
    main()
