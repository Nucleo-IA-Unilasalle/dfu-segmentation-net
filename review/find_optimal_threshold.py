import json
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def load_results(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return []
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            return data.get('results', [])
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return []

def analyze_thresholds():
    parser = argparse.ArgumentParser(description="Analyze Optimal Threshold")
    parser.add_argument("--arch", type=str, default="cnn", choices=["cnn", "vit"], help="Architecture analyzed (default: cnn)")
    args = parser.parse_args()
    
    print(f"Analyzing Thresholds for Architecture: {args.arch.upper()}")
    
    # Input files based on architecture
    fp_file = f'review/skin_lesion_validation_{args.arch}.json'
    tp_file = f'review/azh_wound_validation_{args.arch}.json'
    
    # Fallback for backward compatibility if arch-specific files don't exist yet but generic ones do (mostly for 'cnn')
    if args.arch == 'cnn' and not os.path.exists(fp_file) and os.path.exists('review/skin_lesion_validation.json'):
        fp_file = 'review/skin_lesion_validation.json'
    if args.arch == 'cnn' and not os.path.exists(tp_file) and os.path.exists('review/azh_wound_validation.json'):
        tp_file = 'review/azh_wound_validation.json'

    # Load data
    # Non-wounds (Negative Class)
    fp_results = load_results(fp_file)
    # Wounds (Positive Class)
    tp_results = load_results(tp_file)
    
    if not fp_results or not tp_results:
        print(f"Error: Need validation results for both datasets to find optimal threshold.")
        print(f"Expected files:\n  FP (Negatives): {fp_file}\n  TP (Positives): {tp_file}")
        print(f"Please run review/validate_skin_lesions.py and review/validate_wounds_azh.py with --arch {args.arch} first.")
        return

    print(f"Loaded {len(fp_results)} negative samples (Skin Lesions)")
    print(f"Loaded {len(tp_results)} positive samples (AZH Wounds)")
    
    thresholds = np.arange(0.0, 1.01, 0.01)
    
    precisions = []
    recalls = []
    f1_scores = []
    fpr_rates = []
    
    best_f1 = 0.0
    best_thresh = 0.5
    
    for thresh in thresholds:
        # Positives (Wounds)
        # TP: Confidence > Threshold
        tp = sum(1 for r in tp_results if r['confidence'] > thresh)
        fn = len(tp_results) - tp
        
        # Negatives (Non-Wounds)
        # FP: Confidence > Threshold
        fp = sum(1 for r in fp_results if r['confidence'] > thresh)
        tn = len(fp_results) - fp
        
        # Metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        fpr_rates.append(fpr)
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
    # Print Best Result
    print(f"\n{'='*50}")
    print(f"OPTIMAL THRESHOLD ANALYSIS ({args.arch.upper()})")
    print(f"{'='*50}")
    print(f"Best Threshold (by F1 Score): {best_thresh:.2f}")
    
    # Get metrics for best threshold
    idx = int(best_thresh * 100)
    print(f"  F1 Score:  {f1_scores[idx]:.4f}")
    print(f"  Precision: {precisions[idx]:.4f}")
    print(f"  Recall:    {recalls[idx]:.4f} (Sensitivity)")
    print(f"  FPR:       {fpr_rates[idx]:.4f} (False Positive Rate)")
    
    print(f"\nComparison with Default (0.50):")
    idx_def = 50
    print(f"  F1 Score:  {f1_scores[idx_def]:.4f}")
    print(f"  Precision: {precisions[idx_def]:.4f}")
    print(f"  Recall:    {recalls[idx_def]:.4f}")
    print(f"  FPR:       {fpr_rates[idx_def]:.4f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, label='Precision')
    plt.plot(thresholds, recalls, label='Recall (Sensitivity)')
    plt.plot(thresholds, f1_scores, label='F1 Score')
    plt.plot(thresholds, fpr_rates, label='False Positive Rate', linestyle='--')
    
    plt.axvline(x=best_thresh, color='r', linestyle=':', label=f'Best Thresh ({best_thresh:.2f})')
    
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title(f'Metrics vs Threshold ({args.arch.upper()})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_plot = f'review/threshold_analysis_{args.arch}.png'
    plt.savefig(output_plot)
    print(f"\nAnalysis plot saved to: {output_plot}")

if __name__ == "__main__":
    analyze_thresholds()
