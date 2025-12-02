"""
Comparison script for wound classifier models.
Compares: CNN, Swin, ViT, and ViT 3-channel (ablation) models.
Creates training curves figure and JSON with exact values.
"""

import os
import json
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import numpy as np


# Model configuration
MODELS_CONFIG: Dict[str, Dict[str, str]] = {
    "CNN": {
        "metrics_file": "wound_classifier_metrics.json",
        "display_name": "CNN (4-channel)",
        "color": "#1f77b4"
    },
    "Swin": {
        "metrics_file": "wound_classifier_swin_best_model_metrics.json",
        "display_name": "Swin-B (4-channel)",
        "color": "#ff7f0e"
    },
    "ViT": {
        "metrics_file": "wound_classifier_vit_best_model_metrics.json",
        "display_name": "ViT-B/16 (4-channel)",
        "color": "#2ca02c"
    },
    "ViT_3ch": {
        "metrics_file": "wound_classifier_ablation_3channel_metrics.json",
        "display_name": "ViT-B/16 (3-channel, ablation)",
        "color": "#d62728"
    }
}


def load_model_metrics(base_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load metrics from all model JSON files."""
    all_metrics: Dict[str, Dict[str, Any]] = {}
    
    for model_key, config in MODELS_CONFIG.items():
        metrics_path = os.path.join(base_dir, config["metrics_file"])
        
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            all_metrics[model_key] = metrics
            print(f"✓ Loaded metrics for {config['display_name']}")
        else:
            print(f"✗ Metrics file not found for {config['display_name']}: {metrics_path}")
    
    return all_metrics


def plot_training_curves(
    all_metrics: Dict[str, Dict[str, Any]],
    output_path: str
) -> None:
    """Plot training curves for all models."""
    
    # Check which models have training history
    models_with_history = {
        k: v for k, v in all_metrics.items() 
        if "training_history" in v and v["training_history"]
    }
    
    if not models_with_history:
        print("No models have training history. Skipping training curves plot.")
        return
    
    print(f"\nPlotting training curves for {len(models_with_history)} models...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Wound Classifier Model Comparison - Training Metrics Over Epochs", 
                 fontsize=14, fontweight='bold')
    
    metrics_to_plot = [
        ("train_loss", "Training Loss", axes[0, 0]),
        ("val_loss", "Validation Loss", axes[0, 1]),
        ("train_accuracy", "Training Accuracy (%)", axes[0, 2]),
        ("val_accuracy", "Validation Accuracy (%)", axes[1, 0]),
        ("val_f1", "Validation F1 Score", axes[1, 1]),
        ("val_precision", "Validation Precision", axes[1, 2]),
    ]
    
    for metric_key, metric_label, ax in metrics_to_plot:
        for model_key, metrics in models_with_history.items():
            history = metrics["training_history"]
            if metric_key in history:
                values = history[metric_key]
                epochs = range(1, len(values) + 1)
                config = MODELS_CONFIG[model_key]
                ax.plot(
                    epochs, 
                    values, 
                    label=config["display_name"],
                    color=config["color"],
                    linewidth=2,
                    marker='o',
                    markersize=3
                )
        
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to: {output_path}")


def plot_final_metrics_comparison(
    all_metrics: Dict[str, Dict[str, Any]],
    output_path: str
) -> None:
    """Plot bar chart comparison of final metrics."""
    
    if not all_metrics:
        print("No metrics available for comparison plot.")
        return
    
    print("\nPlotting final metrics comparison...")
    
    # Metrics to compare
    metrics_keys = [
        ("final_accuracy", "Accuracy (%)"),
        ("final_precision", "Precision"),
        ("final_recall", "Recall"),
        ("final_f1", "F1 Score")
    ]
    
    # Prepare data
    model_names = []
    model_colors = []
    for model_key in all_metrics.keys():
        model_names.append(MODELS_CONFIG[model_key]["display_name"])
        model_colors.append(MODELS_CONFIG[model_key]["color"])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Wound Classifier Model Comparison - Final Test Metrics", 
                 fontsize=14, fontweight='bold')
    
    axes_flat = axes.flatten()
    
    for idx, (metric_key, metric_label) in enumerate(metrics_keys):
        ax = axes_flat[idx]
        
        values = []
        for model_key in all_metrics.keys():
            value = all_metrics[model_key].get(metric_key, 0)
            # Convert accuracy to percentage if needed
            if metric_key == "final_accuracy" and value > 1:
                values.append(value)
            elif metric_key == "final_accuracy":
                values.append(value * 100)
            else:
                values.append(value)
        
        x_pos = np.arange(len(model_names))
        bars = ax.bar(x_pos, values, color=model_colors, edgecolor='black', linewidth=1)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.4f}' if metric_key != "final_accuracy" else f'{val:.2f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
        
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Final metrics comparison saved to: {output_path}")


def create_comparison_json(
    all_metrics: Dict[str, Dict[str, Any]],
    output_path: str
) -> Dict[str, Any]:
    """Create comprehensive comparison JSON."""
    
    comparison_data: Dict[str, Any] = {
        "models": {},
        "comparison_summary": {},
        "best_model_per_metric": {}
    }
    
    # Extract data for each model
    for model_key, metrics in all_metrics.items():
        config = MODELS_CONFIG[model_key]
        model_data = {
            "display_name": config["display_name"],
            "final_metrics": {
                "accuracy": metrics.get("final_accuracy", None),
                "precision": metrics.get("final_precision", None),
                "recall": metrics.get("final_recall", None),
                "f1_score": metrics.get("final_f1", None),
                "best_f1_score": metrics.get("best_f1_score", None)
            },
            "detailed_metrics": metrics.get("detailed_metrics", {}),
            "confusion_matrix": metrics.get("confusion_matrix", None),
            "hyperparameters": metrics.get("hyperparameters", {}),
            "training_history": metrics.get("training_history", None)
        }
        comparison_data["models"][model_key] = model_data
    
    # Create comparison summary
    metrics_for_comparison = ["accuracy", "precision", "recall", "f1_score"]
    
    for metric in metrics_for_comparison:
        metric_values: Dict[str, float] = {}
        for model_key, model_data in comparison_data["models"].items():
            value = model_data["final_metrics"].get(metric)
            if value is not None:
                metric_values[model_key] = value
        
        if metric_values:
            best_model = max(metric_values, key=lambda k: metric_values[k])
            comparison_data["best_model_per_metric"][metric] = {
                "model": best_model,
                "display_name": MODELS_CONFIG[best_model]["display_name"],
                "value": metric_values[best_model]
            }
            comparison_data["comparison_summary"][metric] = metric_values
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"Comparison JSON saved to: {output_path}")
    return comparison_data


def print_summary_table(comparison_data: Dict[str, Any]) -> None:
    """Print a summary table of the comparison."""
    
    print("\n" + "=" * 80)
    print("WOUND CLASSIFIER MODEL COMPARISON SUMMARY")
    print("=" * 80)
    
    # Header
    models = list(comparison_data["models"].keys())
    header = f"{'Metric':<20}"
    for model_key in models:
        display_name = MODELS_CONFIG[model_key]["display_name"]
        # Truncate long names
        short_name = display_name[:15] + "..." if len(display_name) > 18 else display_name
        header += f"{short_name:>18}"
    print(header)
    print("-" * 80)
    
    # Metrics rows
    metrics = [
        ("Accuracy (%)", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1 Score", "f1_score")
    ]
    
    for metric_label, metric_key in metrics:
        row = f"{metric_label:<20}"
        best_value = 0
        best_model = None
        
        # Find best for highlighting
        for model_key in models:
            value = comparison_data["models"][model_key]["final_metrics"].get(metric_key)
            if value is not None and value > best_value:
                best_value = value
                best_model = model_key
        
        for model_key in models:
            value = comparison_data["models"][model_key]["final_metrics"].get(metric_key)
            if value is not None:
                if metric_key == "accuracy":
                    formatted_value = f"{value:.2f}%"
                else:
                    formatted_value = f"{value:.4f}"
                
                # Mark best with asterisk
                if model_key == best_model:
                    formatted_value = f"*{formatted_value}"
                row += f"{formatted_value:>18}"
            else:
                row += f"{'N/A':>18}"
        print(row)
    
    print("-" * 80)
    print("* indicates best performance for that metric")
    
    # Print best model summary
    print("\n" + "=" * 80)
    print("BEST MODEL PER METRIC:")
    print("=" * 80)
    for metric, info in comparison_data["best_model_per_metric"].items():
        print(f"  {metric.upper()}: {info['display_name']} ({info['value']:.4f})")
    print("=" * 80 + "\n")


def main() -> None:
    """Main comparison function."""
    # Get base directory (same as script location)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if not base_dir:
        base_dir = "."
    
    print("=" * 60)
    print("WOUND CLASSIFIER MODEL COMPARISON")
    print("=" * 60)
    print(f"\nLooking for metrics files in: {base_dir}\n")
    
    # Load all metrics
    all_metrics = load_model_metrics(base_dir)
    
    if not all_metrics:
        print("\nERROR: No metrics files found!")
        print("Please ensure you have run the training scripts first:")
        print("  - py wound_classifier.py")
        print("  - py wound_classifier_swin.py")
        print("  - py wound_classifier_vit.py")
        print("  - py wound_classifier_ablation_3channel.py")
        return
    
    print(f"\nLoaded {len(all_metrics)} model(s) for comparison")
    
    # Create comparison JSON
    output_json_path = os.path.join(base_dir, "classifier_comparison.json")
    comparison_data = create_comparison_json(all_metrics, output_json_path)
    
    # Plot training curves
    training_curves_path = os.path.join(base_dir, "classifier_training_curves.png")
    plot_training_curves(all_metrics, training_curves_path)
    
    # Plot final metrics comparison
    final_metrics_path = os.path.join(base_dir, "classifier_final_metrics_comparison.png")
    plot_final_metrics_comparison(all_metrics, final_metrics_path)
    
    # Print summary table
    print_summary_table(comparison_data)
    
    print("Comparison complete!")
    print(f"  - Training curves: {training_curves_path}")
    print(f"  - Final metrics comparison: {final_metrics_path}")
    print(f"  - JSON with exact values: {output_json_path}")


if __name__ == "__main__":
    main()

