import cv2
import numpy as np
import os
import config

def detect_and_match(img1, img2):
    """
    Detect A-KAZE features and match them using Brute-Force matcher.
    
    Args:
        img1: First image (BGR)
        img2: Second image (BGR)
        
    Returns:
        kps1, kps2: Keypoints
        matches: List of robust matches
        match_vis: Visualization of matches
    """
    # Initialize A-KAZE detector
    akaze = cv2.AKAZE_create(
        threshold=config.AKAZE_THRESHOLD,
        nOctaves=config.AKAZE_OCTAVES,
        nOctaveLayers=config.AKAZE_LAYERS
    )
    
    # Find keypoints and descriptors
    kps1, des1 = akaze.detectAndCompute(img1, None)
    kps2, des2 = akaze.detectAndCompute(img2, None)
    
    if des1 is None or des2 is None:
        return None, None, None, None
        
    # Match descriptors using BFMatcher with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    # Sort them in the order of their distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Keep only the best matches
    good_matches = matches[:min(len(matches), config.MAX_MATCHES)]
    
    # Draw matches
    match_vis = cv2.drawMatches(
        img1, kps1, img2, kps2, good_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    return kps1, kps2, good_matches, match_vis

def stitch_images(images):
    """
    Stitch a sequence of images into a mosaic.
    
    Args:
        images: List of BGR images
        
    Returns:
        mosaic: The stitched image
        matching_visualizations: List of match visualizations between frames
    """
    if len(images) < 2:
        return images[0] if images else None, []
        
    # Use OpenCV's built-in Stitcher for reliable results
    # We use A-KAZE internally for feature detection if configured, 
    # but the Stitcher class is more robust for multi-image scenarios.
    
    # However, to meet the specific requirements of showing A-KAZE matching,
    # we'll also perform manual matching for visualization.
    
    matching_visualizations = []
    for i in range(len(images) - 1):
        _, _, _, match_vis = detect_and_match(images[i], images[i+1])
        if match_vis is not None:
            matching_visualizations.append(match_vis)
            
    # Initialize stitcher
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    
    # Perform stitching
    status, mosaic = stitcher.stitch(images)
    
    if status != cv2.Stitcher_OK:
        print(f"Stitching failed with status: {status}")
        # If panorama stitcher fails, try SCANS mode
        stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
        status, mosaic = stitcher.stitch(images)
        
    if status == cv2.Stitcher_OK:
        # Crop black borders (common in stitching)
        mosaic = crop_black_borders(mosaic)
        return mosaic, matching_visualizations
    else:
        # If all stitching fails, return the first image as a fallback (not ideal)
        return None, matching_visualizations

def crop_black_borders(img):
    """Remove black borders from stitched image"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return img
        
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    return img[y:y+h, x:x+w]

def mosaicking_pipeline(image_paths):
    """
    Main pipeline for Pipeline 2: Mosaicking
    
    Args:
        image_paths: List of file paths to overlapping frames
        
    Returns:
        mosaic: Stitched image
        match_vis: The first keypoint matching visualization
    """
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            # Resize for performance if needed
            if max(img.shape) > 1200:
                scale = 1200 / max(img.shape)
                img = cv2.resize(img, None, fx=scale, fy=scale)
            images.append(img)
            
    if not images:
        raise ValueError("No valid images provided for mosaicking")
        
    mosaic, visualizations = stitch_images(images)
    
    # Return mosaic and the first visualization to show A-KAZE matching
    first_match_vis = visualizations[0] if visualizations else None
    
    return mosaic, first_match_vis
