import os
import json
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import kagglehub
import sys
from typing import List, Dict, Any, Tuple, Optional
import argparse

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pretrained import EfficientNetUNet

# ImageNet normalization stats (used for transformer-based models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Architecture configurations
ARCH_CONFIG: Dict[str, Dict[str, Any]] = {
    "cnn": {
        "module": "wound_classifier",
        "class_name": "WoundVerificationModel",
        "model_path": "wound_classifier_best_model.pth",
        "image_size": 256,
        "uses_mask": True,
        "uses_normalization": False,
    },
    "vit": {
        "module": "wound_classifier_vit",
        "class_name": "WoundVerificationViT",
        "model_path": "wound_classifier_vit_best_model.pth",
        "image_size": 224,
        "uses_mask": True,
        "uses_normalization": True,
    },
    "swin": {
        "module": "wound_classifier_swin",
        "class_name": "WoundVerificationSwin",
        "model_path": "wound_classifier_swin_best_model.pth",
        "image_size": 224,
        "uses_mask": True,
        "uses_normalization": True,
    },
    "cnn_ablation": {
        "module": "wound_classifier_cnn_ablation_3channel",
        "class_name": "WoundVerificationModel",
        "model_path": "wound_classifier_cnn_ablation_3channel_best_model.pth",
        "image_size": 256,
        "uses_mask": False,  # True 3-channel ablation: RGB only, no mask
        "uses_normalization": False,
    },
    "vit_ablation": {
        "module": "wound_classifier_ablation_3channel",
        "class_name": "WoundVerificationViT",
        "model_path": "wound_classifier_ablation_3channel_best_model.pth",
        "image_size": 224,
        "uses_mask": False,  # Only RGB input, no mask
        "uses_normalization": True,
    },
}


def load_models(
    segmentation_model_path: str,
    classifier_model_path: str,
    device: torch.device,
    architecture: str
) -> Tuple[Optional[nn.Module], nn.Module]:
    """Load both segmentation and verification models."""
    config = ARCH_CONFIG[architecture]
    
    # Load segmentation model (only if architecture uses mask)
    seg_model: Optional[nn.Module] = None
    if config["uses_mask"]:
    seg_model = EfficientNetUNet(out_channels=1, pretrained=False).to(device)
    seg_model.load_state_dict(torch.load(segmentation_model_path, map_location=device))
    seg_model.eval()
    
    # Load verification model based on architecture
    try:
        module = __import__(config["module"])
        model_class = getattr(module, config["class_name"])
        verif_model = model_class().to(device)
    except ImportError as e:
        print(f"Error: Could not import {config['class_name']} from {config['module']}.")
        print(f"Details: {e}")
            sys.exit(1)
        
    verif_model.load_state_dict(torch.load(classifier_model_path, map_location=device))
    verif_model.eval()
    
    return seg_model, verif_model

def predict_with_verification(
    image_path: str,
    segmentation_model: Optional[nn.Module],
    verification_model: nn.Module,
    device: torch.device,
    image_size: int,
    uses_mask: bool,
    uses_normalization: bool,
    seg_threshold: float = 0.5,
    verif_threshold: float = 0.5
) -> Tuple[bool, float, float]:
    """
    Run full prediction pipeline.
    Returns: is_wound, confidence, seg_percentage
    """
    try:
        # Build transforms list
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
        if uses_normalization:
            transform_list.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
        
        transform = transforms.Compose(transform_list)
        
        # Segmentation model always uses non-normalized input
        seg_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        if uses_mask:
            # Step 1: Run segmentation (always with non-normalized input)
            seg_image_tensor = seg_transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
                mask_pred = segmentation_model(seg_image_tensor)
        
        mask_np = mask_pred.squeeze().cpu().numpy()
        
        # Calculate segmentation percentage
        total_pixels = mask_np.size
        segmented_pixels = (mask_np > seg_threshold).sum()
        seg_percentage = (segmented_pixels / total_pixels) * 100.0
        
            # Step 2: Run verification with mask
        mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0).to(device)
        combined_input = torch.cat([image_tensor, mask_tensor], dim=1)
        seg_percentage_tensor = torch.tensor([[seg_percentage]], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            logits = verification_model(combined_input, seg_percentage_tensor)
            confidence = torch.sigmoid(logits).item()
        else:
            # Ablation model: RGB only, no mask
            seg_percentage = 0.0
            with torch.no_grad():
                logits = verification_model(image_tensor)
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
    arch_choices = list(ARCH_CONFIG.keys())
    parser = argparse.ArgumentParser(description="Validate Skin Lesion Robustness")
    parser.add_argument("--arch", type=str, default="cnn", choices=arch_choices, 
                        help=f"Model architecture to use (default: cnn). Options: {arch_choices}")
    parser.add_argument("--model_path", type=str, default=None, 
                        help="Path to verification model weights (optional, defaults based on arch)")
    parser.add_argument("--threshold", type=float, default=0.5, 
                        help="Verification threshold (default: 0.5)")
    args = parser.parse_args()

    config = ARCH_CONFIG[args.arch]
    print(f"Starting Skin Lesion Robustness Validation ({args.arch.upper()})...")
    
    # Configuration
    seg_model_path = "pretrained_best_efficientnet_b4_unet_model.pth"
    verif_model_path = args.model_path if args.model_path else config["model_path"]
    image_size = config["image_size"]
    uses_mask = config["uses_mask"]
    uses_normalization = config["uses_normalization"]
    
    output_file = f"review/skin_lesion_validation_{args.arch}.json"
    
    print(f"Configuration:")
    print(f"  Architecture: {args.arch}")
    print(f"  Verification Model: {verif_model_path}")
    print(f"  Image Size: {image_size}")
    print(f"  Uses Mask: {uses_mask}")
    print(f"  Uses Normalization: {uses_normalization}")
    print(f"  Threshold: {args.threshold}")
    
    # Check models exist
    if uses_mask and not os.path.exists(seg_model_path):
        print(f"Error: Segmentation model not found: {seg_model_path}")
        print("Please train the segmentation model first.")
        return
    if not os.path.exists(verif_model_path):
        print(f"Error: Verification model not found: {verif_model_path}")
        print("Please train the verification model first.")
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
            image_path=img_path, 
            segmentation_model=seg_model, 
            verification_model=verif_model, 
            device=device, 
            image_size=image_size, 
            uses_mask=uses_mask,
            uses_normalization=uses_normalization,
            verif_threshold=args.threshold
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
