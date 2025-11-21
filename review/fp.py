import os
import json
import time
from typing import Dict, List, Tuple, Any
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import kagglehub
from datasets import load_dataset

# Import model from parent directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pretrained import EfficientNetUNet


class NoMaskDataset(Dataset):
    """Dataset class for loading images without ground truth masks (for false positive testing)."""
    
    def __init__(self, image_paths: List[str], transform: transforms.Compose = None):
        """
        Args:
            image_paths: List of full paths to images
            transform: Optional transform to apply to images
        """
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Returns:
            image_tensor: Transformed image tensor
            image_path: Path to the image file
        """
        img_path = self.image_paths[idx]
        
        # Load image and convert to RGB
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, img_path


def download_miniimagenet(download_dir: str) -> str:
    """
    Download MiniImageNet dataset from Kaggle.
    
    Args:
        download_dir: Directory to store downloaded dataset
        
    Returns:
        Path to the dataset directory
    """
    print("Downloading MiniImageNet dataset from Kaggle...")
    path = kagglehub.dataset_download("deeptrial/miniimagenet")
    print(f"MiniImageNet downloaded to: {path}")
    return path


def download_skin_cancer(download_dir: str) -> str:
    """
    Download Skin Cancer dataset from HuggingFace.
    
    Args:
        download_dir: Directory to store downloaded dataset
        
    Returns:
        Path to the dataset directory
    """
    print("Downloading Skin Cancer dataset from HuggingFace...")
    dataset = load_dataset("Pranavkpba2000/skin_cancer_small_dataset", split="test")
    
    # Save images locally
    save_dir = os.path.join(download_dir, "skin_cancer")
    os.makedirs(save_dir, exist_ok=True)
    
    image_paths = []
    for idx, item in enumerate(dataset):
        image_path = os.path.join(save_dir, f"image_{idx}.jpg")
        if not os.path.exists(image_path):
            item['image'].save(image_path)
        image_paths.append(image_path)
    
    print(f"Skin Cancer dataset saved to: {save_dir}")
    print(f"Number of images: {len(image_paths)}")
    return save_dir


def download_skin_disease(download_dir: str) -> str:
    """
    Download Skin Disease dataset from Kaggle.
    
    Args:
        download_dir: Directory to store downloaded dataset
        
    Returns:
        Path to the dataset directory
    """
    print("Downloading Skin Disease dataset from Kaggle...")
    path = kagglehub.dataset_download("pacificrm/skindiseasedataset")
    print(f"Skin Disease dataset downloaded to: {path}")
    return path


def get_image_paths_from_directory(directory: str, extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')) -> List[str]:
    """
    Recursively get all image paths from a directory.
    
    Args:
        directory: Root directory to search
        extensions: Tuple of valid image extensions
        
    Returns:
        List of full paths to image files
    """
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extensions):
                image_paths.append(os.path.join(root, file))
    return sorted(image_paths)


def calculate_false_positive_metrics(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    image_size: int = 256
) -> Dict[str, Any]:
    """
    Calculate false positive metrics on a dataset without wound masks.
    
    Args:
        model: Trained segmentation model
        dataloader: DataLoader with images (no masks)
        device: Device to run inference on
        threshold: Threshold for binary segmentation
        image_size: Size of images (for calculating total pixels)
        
    Returns:
        Dictionary containing all false positive metrics
    """
    model.eval()
    
    # Metrics accumulators
    total_images = 0
    images_with_fp = 0
    images_with_fp_above_1_percent = 0
    images_with_fp_above_5_percent = 0
    total_segmented_pixels = 0
    total_pixels = 0
    max_segmented_pixels = 0
    max_segmented_area_percent = 0.0
    
    per_image_results = []
    
    with torch.no_grad():
        for batch_idx, (images, image_paths) in enumerate(dataloader):
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Apply threshold to get binary predictions
            predictions = (outputs > threshold).float()
            
            # Process each image in the batch
            for i in range(predictions.shape[0]):
                pred_mask = predictions[i].cpu().numpy()
                
                # Calculate metrics for this image
                segmented_pixels = int(pred_mask.sum())
                total_image_pixels = pred_mask.size
                segmented_area_percent = (segmented_pixels / total_image_pixels) * 100.0
                
                # Update accumulators
                total_images += 1
                total_segmented_pixels += segmented_pixels
                total_pixels += total_image_pixels
                
                # Count images with any false positives
                if segmented_pixels > 0:
                    images_with_fp += 1
                
                # Count images with FP above thresholds
                if segmented_area_percent > 1.0:
                    images_with_fp_above_1_percent += 1
                if segmented_area_percent > 5.0:
                    images_with_fp_above_5_percent += 1
                
                # Update max segmented area
                if segmented_pixels > max_segmented_pixels:
                    max_segmented_pixels = segmented_pixels
                    max_segmented_area_percent = segmented_area_percent
                
                # Store per-image results
                per_image_results.append({
                    "image_path": image_paths[i],
                    "segmented_pixels": segmented_pixels,
                    "segmented_area_percent": round(segmented_area_percent, 4)
                })
            
            # Progress update
            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx}/{len(dataloader)}")
    
    # Calculate final metrics
    image_level_fp_rate = (images_with_fp / total_images) * 100.0 if total_images > 0 else 0.0
    pixel_level_fp_rate = (total_segmented_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0
    avg_segmented_pixels = total_segmented_pixels / total_images if total_images > 0 else 0.0
    avg_segmented_area_percent = (avg_segmented_pixels / (image_size * image_size)) * 100.0
    
    metrics = {
        "total_images": total_images,
        "image_level_metrics": {
            "images_with_fp": images_with_fp,
            "fp_rate_percent": round(image_level_fp_rate, 4),
            "images_with_fp_above_1_percent": images_with_fp_above_1_percent,
            "images_with_fp_above_5_percent": images_with_fp_above_5_percent
        },
        "pixel_level_metrics": {
            "total_pixels": int(total_pixels),
            "segmented_pixels": int(total_segmented_pixels),
            "fp_rate_percent": round(pixel_level_fp_rate, 4),
            "avg_segmented_pixels_per_image": round(avg_segmented_pixels, 2),
            "avg_segmented_area_percent": round(avg_segmented_area_percent, 4),
            "max_segmented_pixels": int(max_segmented_pixels),
            "max_segmented_area_percent": round(max_segmented_area_percent, 4)
        },
        "per_image_results": per_image_results
    }
    
    return metrics


def load_model(model_path: str, device: torch.device) -> nn.Module:
    """
    Load the trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint file
        device: Device to load model on
        
    Returns:
        Loaded model in evaluation mode
    """
    model = EfficientNetUNet(out_channels=1, pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def evaluate_dataset(
    model: nn.Module,
    dataset_name: str,
    dataset_source: str,
    image_paths: List[str],
    device: torch.device,
    threshold: float = 0.5,
    batch_size: int = 8,
    image_size: int = 256
) -> Dict[str, Any]:
    """
    Evaluate model on a dataset and calculate false positive metrics.
    
    Args:
        model: Trained segmentation model
        dataset_name: Name of the dataset
        dataset_source: Source URL of the dataset
        image_paths: List of paths to images
        device: Device to run inference on
        threshold: Threshold for binary segmentation
        batch_size: Batch size for evaluation
        image_size: Size to resize images to
        
    Returns:
        Dictionary containing all metrics and metadata
    """
    print(f"\n{'='*60}")
    print(f"Evaluating dataset: {dataset_name}")
    print(f"{'='*60}")
    print(f"Number of images: {len(image_paths)}")
    
    # Create dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    dataset = NoMaskDataset(image_paths, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Calculate metrics
    start_time = time.time()
    metrics = calculate_false_positive_metrics(model, dataloader, device, threshold, image_size)
    evaluation_time = time.time() - start_time
    
    # Build final results
    results = {
        "dataset_name": dataset_name,
        "dataset_source": dataset_source,
        "threshold": threshold,
        "evaluation_time_seconds": round(evaluation_time, 2),
        **metrics
    }
    
    # Print summary
    print(f"\nEvaluation Summary:")
    print(f"  Total images: {results['total_images']}")
    print(f"  Images with FP: {results['image_level_metrics']['images_with_fp']} ({results['image_level_metrics']['fp_rate_percent']}%)")
    print(f"  Images with FP > 1%: {results['image_level_metrics']['images_with_fp_above_1_percent']}")
    print(f"  Images with FP > 5%: {results['image_level_metrics']['images_with_fp_above_5_percent']}")
    print(f"  Pixel-level FP rate: {results['pixel_level_metrics']['fp_rate_percent']}%")
    print(f"  Avg segmented area: {results['pixel_level_metrics']['avg_segmented_area_percent']}%")
    print(f"  Max segmented area: {results['pixel_level_metrics']['max_segmented_area_percent']}%")
    print(f"  Evaluation time: {evaluation_time:.2f}s")
    
    return results


def save_results_to_json(results: Dict[str, Any], output_path: str) -> None:
    """
    Save evaluation results to JSON file.
    
    Args:
        results: Dictionary containing evaluation results
        output_path: Path to save JSON file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


def main() -> None:
    """Main function to evaluate false positives on all datasets."""
    # Configuration
    model_path = "pretrained_best_efficientnet_b4_unet_model.pth"
    download_dir = "datasets"
    output_dir = "review"
    threshold = 0.5
    batch_size = 8
    image_size = 256
    
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = load_model(model_path, device)
    print("Model loaded successfully!")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(download_dir, exist_ok=True)
    
    # Dataset configurations
    datasets_config = [
        {
            "name": "MiniImageNet",
            "source": "https://www.kaggle.com/datasets/deeptrial/miniimagenet",
            "download_func": download_miniimagenet,
            "output_file": "fp_results_miniimagenet.json"
        },
        {
            "name": "Skin Cancer",
            "source": "https://huggingface.co/datasets/Pranavkpba2000/skin_cancer_small_dataset",
            "download_func": download_skin_cancer,
            "output_file": "fp_results_skin_cancer.json"
        },
        {
            "name": "Skin Disease",
            "source": "https://www.kaggle.com/datasets/pacificrm/skindiseasedataset",
            "download_func": download_skin_disease,
            "output_file": "fp_results_skin_disease.json"
        }
    ]
    
    # Evaluate each dataset
    all_results = []
    for config in datasets_config:
        try:
            # Download dataset
            dataset_path = config["download_func"](download_dir)
            
            # Get all image paths
            image_paths = get_image_paths_from_directory(dataset_path)
            
            if len(image_paths) == 0:
                print(f"Warning: No images found in {dataset_path}")
                continue
            
            # Evaluate
            results = evaluate_dataset(
                model=model,
                dataset_name=config["name"],
                dataset_source=config["source"],
                image_paths=image_paths,
                device=device,
                threshold=threshold,
                batch_size=batch_size,
                image_size=image_size
            )
            
            # Save results
            output_path = os.path.join(output_dir, config["output_file"])
            save_results_to_json(results, output_path)
            
            all_results.append(results)
            
        except Exception as e:
            print(f"Error processing {config['name']}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    for result in all_results:
        print(f"\n{result['dataset_name']}:")
        print(f"  Images: {result['total_images']}")
        print(f"  Image-level FP rate: {result['image_level_metrics']['fp_rate_percent']}%")
        print(f"  Pixel-level FP rate: {result['pixel_level_metrics']['fp_rate_percent']}%")
    
    print(f"\nEvaluation completed! All results saved to '{output_dir}/' directory.")


if __name__ == "__main__":
    main()

