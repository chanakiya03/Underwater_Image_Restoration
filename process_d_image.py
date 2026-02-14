"""
Quick batch processor - processes all images in backend/d_image folder
"""

import os
import cv2
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from sea_thru import sea_thru_pipeline
from depth_estimation import DepthEstimator

def main():
    # Paths
    input_folder = r"backend\d_image"
    output_folder = r"backend\d_image_enhanced"
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    print("=" * 60)
    print("  Processing underwater images...")
    print("=" * 60)
    print(f"\nInput:  {input_folder}")
    print(f"Output: {output_folder}\n")
    
    # Initialize depth estimator
    print("Loading depth estimation model...")
    depth_estimator = DepthEstimator()
    print("[OK] Model loaded\n")
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(input_folder).glob(f'*{ext}'))
        image_files.extend(Path(input_folder).glob(f'*{ext.upper()}'))
    
    total = len(image_files)
    print(f"Found {total} images\n")
    
    if total == 0:
        print(f"[ERROR] No images found in {input_folder}")
        print("   Make sure your images are in backend/d_image/")
        return
    
    # Process each image
    for i, img_path in enumerate(image_files, 1):
        print(f"[{i}/{total}] {img_path.name}...", end=" ")
        
        
        try:
            # Get image path as string
            img_path_str = str(img_path)
            
            # Estimate depth  
            depth = depth_estimator.estimate(img_path_str)
            
            # Read image for processing
            img = cv2.imread(img_path_str)
            if img is None:
                print("[FAIL] Failed to read")
                continue
            
            # Apply Sea-Thru enhancement
            enhanced = sea_thru_pipeline(img, depth)
            
            # Save result
            output_path = os.path.join(output_folder, img_path.name)
            cv2.imwrite(output_path, enhanced)
            
            print("[OK]")
            
        except Exception as e:
            print(f"[ERROR] {str(e)}")
    
    print(f"\n" + "=" * 60)
    print(f"[COMPLETE] Enhanced images saved to:")
    print(f"   {os.path.abspath(output_folder)}")
    print("=" * 60)

if __name__ == "__main__":
    main()
