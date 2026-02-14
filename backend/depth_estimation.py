import torch
import cv2
import numpy as np

class DepthEstimator:
    def __init__(self, model_type="MiDaS_small"):
        """
        Initialize MiDaS depth estimator
        
        Args:
            model_type: "MiDaS_small", "DPT_Large", or "DPT_Hybrid"
        """
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f"Using device: {self.device}")
        
        print(f"Loading MiDaS model: {model_type}...")
        self.model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        self.model.to(self.device)
        self.model.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform
        
        print("MiDaS model loaded successfully!")

    def estimate(self, img_path):
        """
        Estimate depth map from image
        
        Returns:
            depth_map: Normalized depth map [0, 1] where 0 = far, 1 = close
        """
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        input_batch = self.transform(img_rgb).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        
        # MiDaS outputs inverse depth (disparity): higher values = closer objects
        # For Sea-Thru, we need actual depth: higher values = farther objects
        # So we invert it
        depth_map = depth_map.max() - depth_map
        
        # Normalize to [0, 1] where 0 = closest, 1 = farthest
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        if depth_max - depth_min > 1e-6:
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        else:
            # Uniform depth (e.g., flat image)
            depth_map = np.ones_like(depth_map) * 0.5
        
        return depth_map
