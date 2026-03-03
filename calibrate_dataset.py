"""
Calibration script for optimizing Sea-Thru parameters on the d_image dataset.
Pick samples, run parameter sweeps, and generate comparison grids.
"""

import os
import cv2
import numpy as np
import sys
from pathlib import Path
import random

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from sea_thru import sea_thru_pipeline
from depth_estimation import DepthEstimator
import backend.config as config

def calibrate(input_folder, output_folder, num_samples=1):
    """
    Run calibration on selected samples from the dataset.
    """
    print(f"Starting calibration in {input_folder}...")
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize depth estimator
    print("Initializing DepthEstimator...")
    depth_estimator = DepthEstimator()
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(Path(input_folder).glob(f'*{ext}')))
    
    if not image_files:
        print(f"No images found in {input_folder}")
        return

    print(f"Found {len(image_files)} images. Selecting {num_samples} samples...")
    samples = random.sample(image_files, min(num_samples, len(image_files)))
    
    sweep_params = [
        {'name': 'Default_Refined', 'sat': 1.25, 'red': 0.35},
        {'name': 'Vibrant_Red', 'sat': 1.4, 'red': 0.5},
        {'name': 'Natural_Soft', 'sat': 1.1, 'red': 0.25},
    ]

    for img_path in samples:
        img_name = img_path.name
        print(f"\nCalibrating image: {img_name}")
        
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        depth = depth_estimator.estimate(str(img_path))
        
        results = []
        for p in sweep_params:
            print(f"  Testing {p['name']}...")
            enhanced = sea_thru_pipeline(img, depth, red_factor=p['red'], sat_boost=p['sat'])
            results.append((p['name'], enhanced))
        
        # Save comparison
        create_comparison_row(img, results, output_folder, img_name)
    
    print(f"\n✅ Calibration complete! Results in: {output_folder}")

def create_comparison_row(original, results, output_folder, filename):
    """Create a horizontal comparison row for one image"""
    h, w = original.shape[:2]
    # Resize for visualization (max height 400)
    scale = 400.0 / h
    new_h, new_w = int(h * scale), int(w * scale)
    
    row_imgs = [cv2.resize(original, (new_w, new_h))]
    labels = ["Original"]
    
    for name, res in results:
        row_imgs.append(cv2.resize(res, (new_w, new_h)))
        labels.append(name)
        
    # Stack images
    combined = np.hstack(row_imgs)
    
    # Add labels
    for i, label in enumerate(labels):
        cv2.putText(combined, label, (i * new_w + 10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
    output_path = os.path.join(output_folder, f"calibration_{filename}")
    cv2.imwrite(output_path, combined)
    print(f"Saved calibration row to {output_path}")

if __name__ == "__main__":
    INPUT = "d:/projects/robust_underwater_app/backend/d_image"
    OUTPUT = "d:/projects/robust_underwater_app/calibration_results"
    calibrate(INPUT, OUTPUT)
