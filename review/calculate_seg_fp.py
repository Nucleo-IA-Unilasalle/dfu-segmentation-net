import json
import os
import glob

def calculate_seg_stats(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    total_images = data['total_images']
    results = data['per_image_results']
    
    fp_0 = 0
    fp_1 = 0
    fp_5 = 0
    
    for r in results:
        seg_pct = r['segmentation_percentage']
        if seg_pct > 0.0:
            fp_0 += 1
        if seg_pct > 1.0:
            fp_1 += 1
        if seg_pct > 5.0:
            fp_5 += 1
            
    name = data['dataset_name']
    
    print(f"{name}:")
    print(f"  Total: {total_images}")
    print(f"  Seg > 0%: {fp_0} ({fp_0/total_images*100:.2f}%)")
    print(f"  Seg > 1%: {fp_1} ({fp_1/total_images*100:.2f}%)")
    print(f"  Seg > 5%: {fp_5} ({fp_5/total_images*100:.2f}%)")
    print("-" * 40)
    
    return {
        "name": name,
        "total": total_images,
        "fp_0": fp_0,
        "fp_0_rate": fp_0/total_images*100,
        "fp_1": fp_1,
        "fp_1_rate": fp_1/total_images*100,
        "fp_5": fp_5,
        "fp_5_rate": fp_5/total_images*100
    }

def main():
    files = [
        "review/verification_results_miniimagenet.json",
        "review/verification_results_skin_cancer.json",
        "review/verification_results_skin_disease.json"
    ]
    
    all_stats = []
    for f in files:
        if os.path.exists(f):
            stats = calculate_seg_stats(f)
            all_stats.append(stats)
        else:
            print(f"File not found: {f}")

if __name__ == "__main__":
    main()

