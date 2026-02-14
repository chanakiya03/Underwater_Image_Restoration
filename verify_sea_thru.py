"""
Sea-Thru Algorithm Verification Script

This script tests the Sea-Thru implementation with sample underwater images.
It generates enhanced images and depth maps for visual inspection.
"""

import cv2
import os
import sys
import numpy as np
from backend.depth_estimation import DepthEstimator
from backend.sea_thru import sea_thru_pipeline

def create_comparison_image(original, enhanced, depth_vis, output_path):
    """Create a side-by-side comparison image"""
    h, w = original.shape[:2]
    
    # Resize depth map to match
    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_VIRIDIS)
    
    # Create comparison grid: Original | Enhanced | Depth
    comparison = np.zeros((h, w * 3, 3), dtype=np.uint8)
    comparison[:, :w] = original
    comparison[:, w:w*2] = enhanced
    comparison[:, w*2:] = depth_colored
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, 'Original', (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(comparison, 'Enhanced', (w + 10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(comparison, 'Depth Map', (w * 2 + 10, 30), font, 1, (255, 255, 255), 2)
    
    cv2.imwrite(output_path, comparison)
    print(f"Comparison saved to {output_path}")

def main():
    # Check for sample images in various locations
    sample_paths = [
        r"d:\projects\underwater_enhancement\image.png",
        r"d:\projects\under_water\sample.jpg",
        r"d:\projects\c_under_water\test_image.png"
    ]
    
    input_img_path = None
    for path in sample_paths:
        if os.path.exists(path):
            input_img_path = path
            break
    
    if not input_img_path:
        print("⚠️ No sample underwater image found.")
        print("\nTo test the Sea-Thru algorithm:")
        print("1. Download an underwater image (e.g., from Unsplash)")
        print("2. Save it to one of these locations:")
        for path in sample_paths:
            print(f"   - {path}")
        print("\nOr run the web app with: cd backend && python main.py")
        return

    output_dir = "test_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Sea-Thru Algorithm Verification")
    print("=" * 60)
    print(f"Input image: {input_img_path}")
    print(f"Output directory: {output_dir}")
    print()

    try:
        print("[1/3] Initializing depth estimator...")
        estimator = DepthEstimator()
        
        print("[2/3] Estimating depth...")
        depth_map = estimator.estimate(input_img_path)
        
        print("[3/3] Running Sea-Thru pipeline...")
        img = cv2.imread(input_img_path)
        enhanced = sea_thru_pipeline(img, depth_map)
        
        # Save individual outputs
        output_enhanced = os.path.join(output_dir, "enhanced.png")
        cv2.imwrite(output_enhanced, enhanced)
        print(f"\n✅ Enhanced image saved to {output_enhanced}")
        
        depth_vis = (depth_map * 255).astype(np.uint8)
        output_depth = os.path.join(output_dir, "depth_map.png") 
        cv2.imwrite(output_depth, depth_vis)
        print(f"✅ Depth map saved to {output_depth}")
        
        # Create comparison
        output_comparison = os.path.join(output_dir, "comparison.png")
        create_comparison_image(img, enhanced, depth_vis, output_comparison)
        
        print("\n" + "=" * 60)
        print("Verification Complete!")
        print("=" * 60)
        print(f"\nResults saved to '{output_dir}/' directory")
        print("Review the images to assess color correction quality.")

    except Exception as e:
        print(f"\n❌ Error during verification: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
