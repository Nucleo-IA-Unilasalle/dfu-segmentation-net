import os
import json
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Tuple, List, Dict, Any

from pretrained import EfficientNetUNet
from wound_classifier import WoundVerificationModel


def load_models(
    segmentation_model_path: str,
    classifier_model_path: str,
    device: torch.device
) -> Tuple[nn.Module, nn.Module]:
    """
    Load both segmentation and verification models.
    
    Args:
        segmentation_model_path: Path to segmentation model checkpoint
        classifier_model_path: Path to classifier model checkpoint
        device: Device to load models on
        
    Returns:
        Tuple of (segmentation_model, verification_model)
    """
    # Load segmentation model
    seg_model = EfficientNetUNet(out_channels=1, pretrained=False).to(device)
    seg_model.load_state_dict(torch.load(segmentation_model_path, map_location=device))
    seg_model.eval()
    
    # Load verification model
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
) -> Tuple[bool, float, np.ndarray, float]:
    """
    Run full prediction pipeline: segmentation + verification.
    
    Args:
        image_path: Path to input image
        segmentation_model: Trained segmentation model
        verification_model: Trained verification model
        device: Device to run inference on
        image_size: Size to resize image to
        seg_threshold: Threshold for segmentation binarization
        verif_threshold: Threshold for wound/non-wound classification
        
    Returns:
        Tuple of:
        - is_wound: Boolean indicating if it's a wound
        - confidence: Verification model confidence
        - mask: Predicted segmentation mask
        - seg_percentage: Percentage of pixels segmented
    """
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Step 1: Run segmentation
    segmentation_model.eval()
    with torch.no_grad():
        mask_pred = segmentation_model(image_tensor)
    
    mask_np = mask_pred.squeeze().cpu().numpy()
    
    # Calculate segmentation percentage
    total_pixels = mask_np.size
    segmented_pixels = (mask_np > seg_threshold).sum()
    seg_percentage = (segmented_pixels / total_pixels) * 100.0
    
    # Step 2: Run verification
    # Create 4-channel input (RGB + mask)
    mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0).to(device)
    combined_input = torch.cat([image_tensor, mask_tensor], dim=1)
    seg_percentage_tensor = torch.tensor([[seg_percentage]], dtype=torch.float32).to(device)
    
    verification_model.eval()
    with torch.no_grad():
        logits = verification_model(combined_input, seg_percentage_tensor)
        confidence = torch.sigmoid(logits).item()
    
    is_wound = confidence > verif_threshold
    
    return is_wound, confidence, mask_np, seg_percentage


def get_image_files(directory: str) -> List[str]:
    """
    Get all image files from a directory.
    
    Args:
        directory: Path to directory containing images
        
    Returns:
        List of paths to image files
    """
    image_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.bmp', '.BMP')
    image_paths = []
    
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(image_extensions):
                image_paths.append(os.path.join(root, file))
    
    return sorted(image_paths)


def process_folder(
    folder_path: str,
    segmentation_model: nn.Module,
    verification_model: nn.Module,
    device: torch.device,
    image_size: int = 256,
    seg_threshold: float = 0.5,
    verif_threshold: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Process all images in a folder and return results.
    
    Args:
        folder_path: Path to folder containing images
        segmentation_model: Trained segmentation model
        verification_model: Trained verification model
        device: Device to run inference on
        image_size: Size to resize images to
        seg_threshold: Threshold for segmentation binarization
        verif_threshold: Threshold for wound/non-wound classification
        
    Returns:
        List of dictionaries containing results for each image
    """
    image_paths = get_image_files(folder_path)
    
    if len(image_paths) == 0:
        print(f"Warning: No image files found in {folder_path}")
        return []
    
    print(f"Found {len(image_paths)} images to process")
    
    results = []
    
    for idx, image_path in enumerate(image_paths):
        try:
            print(f"Processing [{idx+1}/{len(image_paths)}]: {os.path.basename(image_path)}")
            
            is_wound, confidence, mask, seg_percentage = predict_with_verification(
                image_path,
                segmentation_model,
                verification_model,
                device,
                image_size=image_size,
                seg_threshold=seg_threshold,
                verif_threshold=verif_threshold
            )
            
            # Calculate additional metrics
            mask_binary = (mask > seg_threshold).astype(np.uint8)
            segmented_pixels = int(mask_binary.sum())
            total_pixels = mask_binary.size
            
            result = {
                "image_path": image_path,
                "image_filename": os.path.basename(image_path),
                "is_wound": bool(is_wound),
                "verification_confidence": round(float(confidence), 4),
                "segmentation_percentage": round(float(seg_percentage), 4),
                "segmented_pixels": segmented_pixels,
                "total_pixels": total_pixels,
                "segmentation_threshold": seg_threshold,
                "verification_threshold": verif_threshold
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            result = {
                "image_path": image_path,
                "image_filename": os.path.basename(image_path),
                "error": str(e)
            }
            results.append(result)
    
    return results


def main() -> None:
    """Run inference on all images in inference_test folder."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run wound detection with verification on folder")
    parser.add_argument(
        "--folder",
        default="inference_test",
        help="Path to folder containing images (default: inference_test)"
    )
    parser.add_argument(
        "--seg_model",
        default="pretrained_best_efficientnet_b4_unet_model.pth",
        help="Path to segmentation model"
    )
    parser.add_argument(
        "--verif_model",
        default="wound_classifier_best_model.pth",
        help="Path to verification model"
    )
    parser.add_argument("--seg_threshold", type=float, default=0.5, help="Segmentation threshold")
    parser.add_argument("--verif_threshold", type=float, default=0.5, help="Verification threshold")
    os.makedirs('metrics', exist_ok=True)
    parser.add_argument(
        "--output_json",
        default="metrics/inference_results.json",
        help="Path to output JSON file (default: metrics/inference_results.json)"
    )
    
    args = parser.parse_args()
    
    # Check if models exist
    if not os.path.exists(args.seg_model):
        raise FileNotFoundError(f"Segmentation model not found: {args.seg_model}")
    if not os.path.exists(args.verif_model):
        raise FileNotFoundError(f"Verification model not found: {args.verif_model}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load models
    print("Loading models...")
    seg_model, verif_model = load_models(args.seg_model, args.verif_model, device)
    print("Models loaded!")
    
    # Process all images in folder
    print(f"\nProcessing images from folder: {args.folder}")
    results = process_folder(
        args.folder,
        seg_model,
        verif_model,
        device,
        seg_threshold=args.seg_threshold,
        verif_threshold=args.verif_threshold
    )
    
    if len(results) == 0:
        print("No images processed. Exiting.")
        return
    
    # Calculate summary statistics
    successful_results = [r for r in results if "error" not in r]
    wounds_detected = sum(1 for r in successful_results if r["is_wound"])
    avg_confidence = np.mean([r["verification_confidence"] for r in successful_results]) if successful_results else 0.0
    avg_seg_percentage = np.mean([r["segmentation_percentage"] for r in successful_results]) if successful_results else 0.0
    
    # Create output dictionary
    output_data = {
        "summary": {
            "total_images": len(results),
            "successful_inferences": len(successful_results),
            "failed_inferences": len(results) - len(successful_results),
            "wounds_detected": wounds_detected,
            "non_wounds": len(successful_results) - wounds_detected,
            "average_verification_confidence": round(float(avg_confidence), 4),
            "average_segmentation_percentage": round(float(avg_seg_percentage), 4),
            "segmentation_threshold": args.seg_threshold,
            "verification_threshold": args.verif_threshold
        },
        "results": results
    }
    
    # Save to JSON
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print("INFERENCE SUMMARY")
    print(f"{'='*60}")
    print(f"Total images processed: {len(results)}")
    print(f"Successful inferences: {len(successful_results)}")
    print(f"Failed inferences: {len(results) - len(successful_results)}")
    print(f"Wounds detected: {wounds_detected}")
    print(f"Non-wounds: {len(successful_results) - wounds_detected}")
    print(f"Average verification confidence: {avg_confidence:.4f}")
    print(f"Average segmentation percentage: {avg_seg_percentage:.4f}")
    print(f"{'='*60}")
    print(f"\nResults saved to: {args.output_json}")


if __name__ == "__main__":
    main()

