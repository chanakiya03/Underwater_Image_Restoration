import cv2
import os
import numpy as np
from backend.sea_thru import sea_thru_pipeline

def main():
    # Use an image from the workspace
    input_img_path = r"d:\projects\underwater_enhancement\image.png"
    if not os.path.exists(input_img_path):
        print(f"Error: Sample image not found at {input_img_path}")
        return

    output_dir = "test_outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_img_path = os.path.join(output_dir, "logic_verified_sea_thru.png")

    print("Running Sea-Thru pipeline with dummy depth map...")
    img = cv2.imread(input_img_path)
    h, w, _ = img.shape
    
    # Create a dummy depth map: top is far (0), bottom is close (1)
    depth_map = np.linspace(0, 1, h).reshape(-1, 1).repeat(w, axis=1)
    
    try:
        enhanced = sea_thru_pipeline(img, depth_map)
        cv2.imwrite(output_img_path, enhanced)
        print(f"Success! Enhanced image saved to {output_img_path}")
    except Exception as e:
        print(f"Error during logic verification: {e}")

if __name__ == "__main__":
    main()
