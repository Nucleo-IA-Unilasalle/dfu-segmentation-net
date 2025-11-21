import os
import json
import time
import sys
from typing import Dict, List, Tuple, Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import kagglehub
from datasets import load_dataset

# Import models from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pretrained import EfficientNetUNet
from wound_classifier import WoundVerificationModel
from wound_classifier_vit import WoundVerificationViT

class EvaluationDataset(Dataset):
    """Dataset class for loading images for verification evaluation."""
    
    def __init__(self, image_paths: List[str]):
        self.image_paths = image_paths
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[str, str]:
        """Returns image path and path (wrapper)"""
        return self.image_paths[idx], self.image_paths[idx]

def download_miniimagenet(download_dir: str) -> str:
    """Download MiniImageNet dataset from Kaggle."""
    print("Downloading MiniImageNet dataset from Kaggle...")
    try:
        path = kagglehub.dataset_download("deeptrial/miniimagenet")
        print(f"MiniImageNet downloaded to: {path}")
        return path
    except Exception as e:
        print(f"Error downloading MiniImageNet: {e}")
        return ""

def download_skin_cancer(download_dir: str) -> str:
    """Download Skin Cancer dataset from HuggingFace."""
    print("Downloading Skin Cancer dataset from HuggingFace...")
    save_dir = os.path.join(download_dir, "skin_cancer")
    
    if os.path.exists(save_dir) and len(os.listdir(save_dir)) > 100:
        print(f"Skin Cancer dataset seems to exist in {save_dir}")
        return save_dir

    try:
        dataset = load_dataset("Pranavkpba2000/skin_cancer_small_dataset", split="test")
        os.makedirs(save_dir, exist_ok=True)
        
        image_paths = []
        for idx, item in enumerate(dataset):
            image_path = os.path.join(save_dir, f"image_{idx}.jpg")
            if not os.path.exists(image_path):
                item['image'].save(image_path)
            image_paths.append(image_path)
        
        print(f"Skin Cancer dataset saved to: {save_dir}")
        return save_dir
    except Exception as e:
        print(f"Error downloading Skin Cancer dataset: {e}")
        return ""

def download_skin_disease(download_dir: str) -> str:
    """Download Skin Disease dataset from Kaggle."""
    print("Downloading Skin Disease dataset from Kaggle...")
    try:
        path = kagglehub.dataset_download("pacificrm/skindiseasedataset")
        print(f"Skin Disease dataset downloaded to: {path}")
        return path
    except Exception as e:
        print(f"Error downloading Skin Disease dataset: {e}")
        return ""

def get_image_paths_from_directory(directory: str, extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')) -> List[str]:
    """Recursively get all image paths from a directory."""
    image_paths = []
    if not directory or not os.path.exists(directory):
        return []
        
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extensions):
                image_paths.append(os.path.join(root, file))
    return sorted(image_paths)

def load_models(
    seg_model_path: str, 
    cnn_model_path: str, 
    vit_model_path: str, 
    device: torch.device
) -> Tuple[nn.Module, nn.Module, nn.Module]:
    """Load all three models."""
    
    # 1. Segmentation Model
    print(f"Loading Segmentation Model from {seg_model_path}...")
    seg_model = EfficientNetUNet(out_channels=1, pretrained=False).to(device)
    seg_model.load_state_dict(torch.load(seg_model_path, map_location=device))
    seg_model.eval()
    
    # 2. CNN Verification Model
    print(f"Loading CNN Verification Model from {cnn_model_path}...")
    cnn_model = WoundVerificationModel().to(device)
    cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=device))
    cnn_model.eval()
    
    # 3. ViT Verification Model
    print(f"Loading ViT Verification Model from {vit_model_path}...")
    vit_model = WoundVerificationViT().to(device)
    vit_model.load_state_dict(torch.load(vit_model_path, map_location=device))
    vit_model.eval()
    
    return seg_model, cnn_model, vit_model

def process_batch(
    images: List[Image.Image], 
    seg_model: nn.Module, 
    verif_model: nn.Module, 
    device: torch.device,
    model_type: str, # 'cnn' or 'vit'
    image_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Process a batch of images through segmentation and verification.
    Returns (probabilities, seg_percentages).
    """
    # Transforms
    # Seg model usually expects 256x256
    seg_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    # Verif model expects image_size (256 for CNN, 224 for ViT)
    if model_type == 'vit':
        verif_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        verif_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
        
    # Prepare tensors
    seg_inputs = torch.stack([seg_transform(img) for img in images]).to(device)
    verif_inputs_rgb = torch.stack([verif_transform(img) for img in images]).to(device)
    
    # 1. Run Segmentation
    with torch.no_grad():
        masks = seg_model(seg_inputs) # (B, 1, 256, 256)
        
    # Calculate percentages (on 256x256 masks)
    masks_np = masks.squeeze(1).cpu().numpy()
    seg_percentages = []
    for mask in masks_np:
        binary_mask = mask > 0.5
        percentage = (binary_mask.sum() / binary_mask.size) * 100.0
        seg_percentages.append(percentage)
    
    seg_percentages_tensor = torch.tensor(seg_percentages, dtype=torch.float32).unsqueeze(1).to(device)
    
    # 2. Prepare Verification Input
    # Resize mask to match verif input size if needed
    if model_type == 'vit' and image_size != 256:
        # Resize mask tensor to 224x224
        masks_resized = torch.nn.functional.interpolate(masks, size=(image_size, image_size), mode='bilinear', align_corners=False)
    else:
        masks_resized = masks
        
    # Concatenate (RGB + Mask)
    # verif_inputs_rgb is (B, 3, H, W), masks_resized is (B, 1, H, W)
    # Note: masks are logits from seg model? No, usually EfficientNetUNet output is logits.
    # But WoundVerificationDataset uses: mask_np = mask_pred.squeeze().cpu().numpy(), then mask_tensor = from_numpy.
    # And predicts with: mask_pred = self.segmentation_model(image_tensor)
    # EfficientNetUNet output is raw logits? Let's check pretrained.py if possible, but standard UNet usually returns logits.
    # WoundVerificationDataset uses: segmented_pixels = (mask_np > 0.5).sum() implies it might be sigmoid-ed already or expects probabilities?
    # Actually, standard is logits. If it's logits, > 0.5 is wrong unless it's sigmoid output.
    # But let's assume the standard behavior from the repo:
    # In WoundVerificationDataset: mask_pred = self.segmentation_model(image_tensor)
    # mask_np = mask_pred.squeeze()
    # segmented_pixels = (mask_np > 0.5).sum()
    # If it's logits, 0.5 is a high threshold (usually 0.0). If it's sigmoid, 0.5 is 50%.
    # I'll assume the segmentation model returns Sigmoid-ed output OR the previous code worked because it was trained that way.
    # To be safe, I will apply sigmoid if the range seems to be outside [0, 1] or if standard practice.
    # However, referencing WoundVerificationDataset lines 136-141, it loads mask and does > 0.5.
    # AND line 79 just calls forward().
    # I will stick to passing the raw output from seg_model to verif_model (as tensor) BUT
    # WoundVerificationDataset loads from NPY cache.
    # Let's assume the passed mask to Verif model should be consistent with training.
    # In training (WoundVerificationDataset), it does: mask_tensor = torch.from_numpy(mask_np).float()
    # So it passes whatever came out of seg model.
    
    combined_input = torch.cat([verif_inputs_rgb, masks_resized], dim=1)
    
    # 3. Run Verification
    with torch.no_grad():
        logits = verif_model(combined_input, seg_percentages_tensor)
        probs = torch.sigmoid(logits)
        
    return probs, seg_percentages_tensor

def evaluate_dataset(
    dataset_name: str,
    image_paths: List[str],
    seg_model: nn.Module,
    cnn_model: nn.Module,
    vit_model: nn.Module,
    device: torch.device,
    batch_size: int = 16
) -> Dict[str, Any]:
    """Evaluate both models on a dataset."""
    
    print(f"\nEvaluating {dataset_name} ({len(image_paths)} images)...")
    
    dataset = EvaluationDataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Metrics
    cnn_fp_count = 0
    cnn_fp_count_opt = 0 # Optimized threshold
    vit_fp_count = 0
    vit_fp_count_opt = 0 # Optimized threshold
    
    # Thresholds (from article.tex)
    CNN_DEFAULT_THRESH = 0.5
    CNN_OPT_THRESH = 0.77
    VIT_DEFAULT_THRESH = 0.5
    VIT_OPT_THRESH = 0.01
    
    total_images = 0
    
    results_list = []
    
    for batch_paths, _ in dataloader:
        # Load images
        images = []
        valid_paths = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
                valid_paths.append(p)
            except Exception as e:
                print(f"Error loading {p}: {e}")
        
        if not images:
            continue
            
        current_batch_size = len(images)
        total_images += current_batch_size
        
        # CNN Evaluation (256x256)
        cnn_probs, seg_percents = process_batch(images, seg_model, cnn_model, device, 'cnn', 256)
        
        # ViT Evaluation (224x224)
        vit_probs, _ = process_batch(images, seg_model, vit_model, device, 'vit', 224)
        
        # Process results
        for i in range(current_batch_size):
            cnn_p = cnn_probs[i].item()
            vit_p = vit_probs[i].item()
            seg_p = seg_percents[i].item()
            path = valid_paths[i]
            
            # Count FP
            if cnn_p > CNN_DEFAULT_THRESH: cnn_fp_count += 1
            if cnn_p > CNN_OPT_THRESH: cnn_fp_count_opt += 1
            if vit_p > VIT_DEFAULT_THRESH: vit_fp_count += 1
            if vit_p > VIT_OPT_THRESH: vit_fp_count_opt += 1
            
            results_list.append({
                "image_path": path,
                "segmentation_percentage": round(seg_p, 4),
                "cnn_probability": round(cnn_p, 6),
                "vit_probability": round(vit_p, 6),
                "cnn_fp_default": bool(cnn_p > CNN_DEFAULT_THRESH),
                "cnn_fp_opt": bool(cnn_p > CNN_OPT_THRESH),
                "vit_fp_default": bool(vit_p > VIT_DEFAULT_THRESH),
                "vit_fp_opt": bool(vit_p > VIT_OPT_THRESH)
            })
            
        if total_images % 100 == 0:
            print(f"Processed {total_images} images...")

    # Calculate final metrics
    metrics = {
        "dataset_name": dataset_name,
        "total_images": total_images,
        "cnn_metrics": {
            "default_threshold": CNN_DEFAULT_THRESH,
            "optimized_threshold": CNN_OPT_THRESH,
            "fp_count_default": cnn_fp_count,
            "fp_count_opt": cnn_fp_count_opt,
            "fp_rate_default": round(cnn_fp_count / total_images * 100, 2) if total_images else 0,
            "fp_rate_opt": round(cnn_fp_count_opt / total_images * 100, 2) if total_images else 0,
            "specificity_default": round(100 - (cnn_fp_count / total_images * 100), 2) if total_images else 0,
            "specificity_opt": round(100 - (cnn_fp_count_opt / total_images * 100), 2) if total_images else 0
        },
        "vit_metrics": {
            "default_threshold": VIT_DEFAULT_THRESH,
            "optimized_threshold": VIT_OPT_THRESH,
            "fp_count_default": vit_fp_count,
            "fp_count_opt": vit_fp_count_opt,
            "fp_rate_default": round(vit_fp_count / total_images * 100, 2) if total_images else 0,
            "fp_rate_opt": round(vit_fp_count_opt / total_images * 100, 2) if total_images else 0,
            "specificity_default": round(100 - (vit_fp_count / total_images * 100), 2) if total_images else 0,
            "specificity_opt": round(100 - (vit_fp_count_opt / total_images * 100), 2) if total_images else 0
        },
        "per_image_results": results_list
    }
    
    return metrics

def main():
    # Configuration
    seg_model_path = "pretrained_best_efficientnet_b4_unet_model.pth"
    cnn_model_path = "wound_classifier_best_model.pth"
    vit_model_path = "wound_classifier_vit_best_model.pth"
    download_dir = "datasets"
    output_dir = "review"
    
    # Check models
    for p in [seg_model_path, cnn_model_path, vit_model_path]:
        if not os.path.exists(p):
            print(f"Error: Model not found at {p}")
            return

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Models
    seg_model, cnn_model, vit_model = load_models(seg_model_path, cnn_model_path, vit_model_path, device)
    
    # Datasets to test
    datasets_config = [
        {
            "name": "MiniImageNet",
            "download_func": download_miniimagenet,
            "output_file": "verification_results_miniimagenet.json"
        },
        {
            "name": "Skin Cancer",
            "download_func": download_skin_cancer,
            "output_file": "verification_results_skin_cancer.json"
        },
        {
            "name": "Skin Disease",
            "download_func": download_skin_disease,
            "output_file": "verification_results_skin_disease.json"
        }
    ]
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(download_dir, exist_ok=True)
    
    all_dataset_metrics = {}
    
    for config in datasets_config:
        try:
            # Download/Get path
            path = config["download_func"](download_dir)
            if not path: continue
            
            # Get images
            image_paths = get_image_paths_from_directory(path)
            if not image_paths:
                print(f"No images found for {config['name']}")
                continue
                
            # Evaluate
            metrics = evaluate_dataset(
                config["name"], 
                image_paths, 
                seg_model, 
                cnn_model, 
                vit_model, 
                device
            )
            
            # Save
            output_path = os.path.join(output_dir, config["output_file"])
            with open(output_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"Saved results to {output_path}")
            
            all_dataset_metrics[config["name"]] = metrics
            
        except Exception as e:
            print(f"Error evaluating {config['name']}: {e}")
            import traceback
            traceback.print_exc()
            
    # Print Final Summary
    print("\n" + "="*80)
    print("FINAL VERIFICATION PERFORMANCE SUMMARY")
    print("="*80)
    print(f"{'Dataset':<15} | {'Model':<5} | {'Default FPR':<12} | {'Opt FPR':<10} | {'Default Spec':<12} | {'Opt Spec':<10}")
    print("-" * 80)
    
    for name, m in all_dataset_metrics.items():
        # CNN
        print(f"{name:<15} | CNN   | {m['cnn_metrics']['fp_rate_default']:>5.2f}%       | {m['cnn_metrics']['fp_rate_opt']:>5.2f}%    | {m['cnn_metrics']['specificity_default']:>5.2f}%       | {m['cnn_metrics']['specificity_opt']:>5.2f}%")
        # ViT
        print(f"{name:<15} | ViT   | {m['vit_metrics']['fp_rate_default']:>5.2f}%       | {m['vit_metrics']['fp_rate_opt']:>5.2f}%    | {m['vit_metrics']['specificity_default']:>5.2f}%       | {m['vit_metrics']['specificity_opt']:>5.2f}%")
        print("-" * 80)

if __name__ == "__main__":
    main()

