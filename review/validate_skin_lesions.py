import os
import json
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import kagglehub
import sys
from typing import List, Dict, Any, Tuple
import argparse

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pretrained import EfficientNetUNet
# We import dynamically based on args now, but keep a default for type hinting if needed
from wound_classifier import WoundVerificationModel as WoundVerificationCNN

def load_models(
    segmentation_model_path: str,
    classifier_model_path: str,
    device: torch.device,
    architecture: str = "cnn"
) -> Tuple[nn.Module, nn.Module]:
    """Load both segmentation and verification models."""
    # Load segmentation model
    seg_model = EfficientNetUNet(out_channels=1, pretrained=False).to(device)
    seg_model.load_state_dict(torch.load(segmentation_model_path, map_location=device))
    seg_model.eval()
    
    # Load verification model based on architecture
    if architecture == "vit":
        try:
            from wound_classifier_vit import WoundVerificationViT
            verif_model = WoundVerificationViT().to(device)
        except ImportError:
            print("Error: Could not import WoundVerificationViT. Make sure wound_classifier_vit.py exists.")
            sys.exit(1)
    else: # default to cnn
        from wound_classifier import WoundVerificationModel
        verif_model = WoundVerificationModel().to(device)
        
    verif_model.load_state_dict(torch.load(classifier_model_path, map_location=device))
    verif_model.eval()
    
    return seg_model, verif_model

def predict_with_verification(
    image_path: str,
    segmentation_model: nn.Module,
    verification_model: nn.Module,
    device: torch.device,
    image_size: int = 256,
    seg_threshold: float = 0.5,
    verif_threshold: float = 0.5
) -> Tuple[bool, float, float]:
    """
    Run full prediction pipeline.
    Returns: is_wound, confidence, seg_percentage
    """
    try:
        # Load and preprocess image
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Step 1: Run segmentation
        with torch.no_grad():
            mask_pred = segmentation_model(image_tensor)
        
        mask_np = mask_pred.squeeze().cpu().numpy()
        
        # Calculate segmentation percentage
        total_pixels = mask_np.size
        segmented_pixels = (mask_np > seg_threshold).sum()
        seg_percentage = (segmented_pixels / total_pixels) * 100.0
        
        # Step 2: Run verification
        mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0).to(device)
        combined_input = torch.cat([image_tensor, mask_tensor], dim=1)
        seg_percentage_tensor = torch.tensor([[seg_percentage]], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            logits = verification_model(combined_input, seg_percentage_tensor)
            confidence = torch.sigmoid(logits).item()
        
        is_wound = confidence > verif_threshold
        return is_wound, confidence, seg_percentage
        
    except Exception as e:
        print(f"Error predicting {image_path}: {e}")
        return False, 0.0, 0.0

def get_image_files(directory: str) -> List[str]:
    """Get all image files from a directory recursively."""
    image_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.bmp')
    image_paths = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(image_extensions):
                image_paths.append(os.path.join(root, file))
    
    return sorted(image_paths)

def main():
    parser = argparse.ArgumentParser(description="Validate Skin Lesion Robustness")
    parser.add_argument("--arch", type=str, default="cnn", choices=["cnn", "vit"], help="Model architecture to use (default: cnn)")
    parser.add_argument("--model_path", type=str, default=None, help="Path to verification model weights (optional, defaults based on arch)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Verification threshold (default: 0.5)")
    args = parser.parse_args()

    print(f"Starting Skin Lesion Robustness Validation ({args.arch.upper()})...")
    
    # Configuration
    seg_model_path = "pretrained_best_efficientnet_b4_unet_model.pth"
    
    # Set default model path based on architecture if not provided
    if args.model_path:
        verif_model_path = args.model_path
    else:
        if args.arch == "vit":
            verif_model_path = "wound_classifier_vit_best_model.pth"
        else:
            verif_model_path = "wound_classifier_best_model.pth"
            
    # Set image size based on architecture
    image_size = 224 if args.arch == "vit" else 256
    
    output_file = f"review/skin_lesion_validation_{args.arch}.json"
    
    print(f"Configuration:")
    print(f"  Architecture: {args.arch}")
    print(f"  Verification Model: {verif_model_path}")
    print(f"  Image Size: {image_size}")
    print(f"  Threshold: {args.threshold}")
    
    # Check models
    if not os.path.exists(seg_model_path) or not os.path.exists(verif_model_path):
        print(f"Error: Models not found.\n  Seg: {seg_model_path}\n  Verif: {verif_model_path}")
        print("Please train the models first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load models
    print("Loading models...")
    seg_model, verif_model = load_models(seg_model_path, verif_model_path, device, architecture=args.arch)
    
    # Download Skin Cancer dataset (mahdavi1202/skin-cancer)
    print("\nDownloading Skin Cancer dataset (mahdavi1202/skin-cancer)...")
    try:
        # Dataset: mahdavi1202/skin-cancer
        dataset_path = kagglehub.dataset_download("mahdavi1202/skin-cancer")
        print(f"Dataset downloaded to: {dataset_path}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return

    # Get images
    image_paths = get_image_files(dataset_path)
    print(f"Found {len(image_paths)} images in dataset.")
    
    # Limit to 500 random images for reasonable validation time
    if len(image_paths) > 500:
        print("Selecting random 500 images for validation...")
        import random
        random.seed(42)
        image_paths = random.sample(image_paths, 500)
    
    # Run validation
    print(f"\nRunning validation on {len(image_paths)} images...")
    print("These are all NON-WOUNDS. Any detection is a False Positive.")
    
    results = []
    false_positives = 0
    
    for idx, img_path in enumerate(image_paths):
        if idx % 50 == 0:
            print(f"Processing {idx}/{len(image_paths)}...")
            
        is_wound, confidence, seg_pct = predict_with_verification(
            img_path, seg_model, verif_model, device, image_size=image_size, verif_threshold=args.threshold
        )
        
        result = {
            "image": os.path.basename(img_path),
            "is_wound": is_wound,
            "confidence": round(confidence, 4),
            "segmentation_pct": round(seg_pct, 2)
        }
        results.append(result)
        
        if is_wound:
            false_positives += 1
            # Print false positives as they happen
            print(f"  [FP] {os.path.basename(img_path)}: Conf={confidence:.4f}, Seg={seg_pct:.2f}%")
            
    # Calculate metrics
    fp_rate = (false_positives / len(image_paths)) * 100.0
    avg_conf = np.mean([r["confidence"] for r in results])
    avg_seg = np.mean([r["segmentation_pct"] for r in results])
    
    summary = {
        "dataset": "Skin Cancer (mahdavi1202)",
        "architecture": args.arch,
        "threshold": args.threshold,
        "total_images": len(image_paths),
        "false_positives": false_positives,
        "false_positive_rate_percent": round(fp_rate, 2),
        "average_confidence": round(float(avg_conf), 4),
        "average_segmentation_pct": round(float(avg_seg), 2)
    }
    
    # Save report
    final_output = {
        "summary": summary,
        "results": results
    }
    
    os.makedirs("review", exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(final_output, f, indent=2)
        
    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY ({args.arch.upper()})")
    print(f"{'='*60}")
    print(f"Dataset: Skin Cancer (mahdavi1202 - Non-Wounds)")
    print(f"Total Images: {len(image_paths)}")
    print(f"False Positives: {false_positives}")
    print(f"False Positive Rate: {fp_rate:.2f}%")
    print(f"Avg Verification Confidence: {avg_conf:.4f}")
    print(f"Avg Segmentation %: {avg_seg:.2f}%")
    print(f"{'='*60}")
    print(f"Detailed results saved to {output_file}")

if __name__ == "__main__":
    main()
