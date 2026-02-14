"""
Batch Processing Script for Underwater Image Enhancement
Process multiple images and test different parameter settings
"""

import os
import cv2
import numpy as np
from pathlib import Path
import sys

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from sea_thru import sea_thru_pipeline
from depth_estimation import DepthEstimator

def process_batch(input_folder, output_folder, params=None):
    """
    Process all images in a folder
    
    Args:
        input_folder: Path to folder with input images
        output_folder: Path to save processed images
        params: Optional dict with custom parameters
    """
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize depth estimator
    print("Loading depth estimation model...")
    depth_estimator = DepthEstimator()
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(input_folder).glob(f'*{ext}'))
        image_files.extend(Path(input_folder).glob(f'*{ext.upper()}'))
    
    print(f"\nFound {len(image_files)} images to process\n")
    
    # Process each image
    for i, img_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] Processing: {img_path.name}")
        
        try:
            # Get image path as string
            img_path_str = str(img_path)
            
            # Estimate depth
            depth = depth_estimator.estimate(img_path_str)
            
            # Read image
            img = cv2.imread(img_path_str)
            if img is None:
                print(f"  [FAIL] Failed to read image")
                continue
            
            # Apply Sea-Thru enhancement
            enhanced = sea_thru_pipeline(img, depth)
            
            # Save result
            output_path = os.path.join(output_folder, f"enhanced_{img_path.name}")
            cv2.imwrite(output_path, enhanced)
            
            print(f"  [OK] Saved to: {output_path}")
            
        except Exception as e:
            print(f"  [ERROR] {str(e)}")
    
    print(f"\n✅ Batch processing complete!")
    print(f"   Output folder: {output_folder}")

def compare_parameters(image_path, output_folder):
    """
    Process one image with different parameter settings for comparison
    
    Args:
        image_path: Path to test image
        output_folder: Folder to save comparison results
    """
    import importlib
    import backend.config as config
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Failed to read image: {image_path}")
        return
    
    print(f"Processing {image_path} with different parameters...\n")
    
    # Initialize depth estimator
    depth_estimator = DepthEstimator()
    depth = depth_estimator.estimate(image_path)
    
    # Define parameter variations to test
    param_sets = [
        {
            'name': 'conservative',
            'SATURATION_BOOST': 1.1,
            'red_factor': 0.3,
            'blue_reduction': 0.2,
            'green_reduction': 0.15
        },
        {
            'name': 'balanced',
            'SATURATION_BOOST': 1.25,
            'red_factor': 0.45,
            'blue_reduction': 0.3,
            'green_reduction': 0.25
        },
        {
            'name': 'vibrant',
            'SATURATION_BOOST': 1.4,
            'red_factor': 0.6,
            'blue_reduction': 0.4,
            'green_reduction': 0.35
        },
        {
            'name': 'aggressive',
            'SATURATION_BOOST': 1.6,
            'red_factor': 0.75,
            'blue_reduction': 0.5,
            'green_reduction': 0.4
        }
    ]
    
    results = []
    
    for params in param_sets:
        print(f"Testing: {params['name']}")
        
        # Update config
        config.SATURATION_BOOST = params['SATURATION_BOOST']
        
        # Process (note: we'd need to modify sea_thru to accept custom params)
        # For now, just use current settings
        enhanced = sea_thru_pipeline(img, depth)
        
        # Save result
        output_path = os.path.join(output_folder, f"{params['name']}_enhanced.jpg")
        cv2.imwrite(output_path, enhanced)
        print(f"  ✅ Saved: {output_path}")
        
        results.append((params['name'], enhanced))
    
    # Create comparison grid
    create_comparison_grid(img, results, output_folder)
    
    print(f"\n✅ Parameter comparison complete!")
    print(f"   Check results in: {output_folder}")

def create_comparison_grid(original, results, output_folder):
    """Create a grid comparing original with all processed versions"""
    
    # Resize all to same height for comparison
    height = 400
    
    # Resize original
    aspect = original.shape[1] / original.shape[0]
    width = int(height * aspect)
    original_resized = cv2.resize(original, (width, height))
    
    # Add label
    labeled_original = original_resized.copy()
    cv2.putText(labeled_original, "ORIGINAL", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    images = [labeled_original]
    
    # Resize and label all results
    for name, img in results:
        resized = cv2.resize(img, (width, height))
        cv2.putText(resized, name.upper(), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        images.append(resized)
    
    # Create grid (2 rows)
    row1 = np.hstack(images[:3]) if len(images) >= 3 else np.hstack(images)
    row2 = np.hstack(images[3:]) if len(images) > 3 else None
    
    if row2 is not None:
        # Pad row2 if needed
        if row2.shape[1] < row1.shape[1]:
            pad_width = row1.shape[1] - row2.shape[1]
            padding = np.zeros((height, pad_width, 3), dtype=np.uint8)
            row2 = np.hstack([row2, padding])
        grid = np.vstack([row1, row2])
    else:
        grid = row1
    
    # Save grid
    grid_path = os.path.join(output_folder, "comparison_grid.jpg")
    cv2.imwrite(grid_path, grid)
    print(f"\n📊 Comparison grid saved: {grid_path}")

if __name__ == "__main__":
    print("=" * 60)
    print("  Underwater Image Enhancement - Batch Processor")
    print("=" * 60)
    print()
    print("Choose an option:")
    print("  1. Process all images in a folder")
    print("  2. Compare different parameters on one image")
    print("  3. Exit")
    print()
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        input_folder = input("\nEnter input folder path: ").strip()
        output_folder = input("Enter output folder path (default: ./batch_output): ").strip()
        
        if not output_folder:
            output_folder = "./batch_output"
        
        if os.path.exists(input_folder):
            process_batch(input_folder, output_folder)
        else:
            print(f"❌ Input folder not found: {input_folder}")
    
    elif choice == "2":
        image_path = input("\nEnter test image path: ").strip()
        output_folder = input("Enter output folder (default: ./comparison_output): ").strip()
        
        if not output_folder:
            output_folder = "./comparison_output"
        
        if os.path.exists(image_path):
            compare_parameters(image_path, output_folder)
        else:
            print(f"❌ Image not found: {image_path}")
    
    else:
        print("Goodbye!")
