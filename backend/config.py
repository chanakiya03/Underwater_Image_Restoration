"""
Configuration settings for the Underwater Image Recovery application
"""

import os

# File upload settings
MAX_FILE_SIZE_MB = 10
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

# Directory settings
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

# MiDaS model settings
MIDAS_MODEL_TYPE = "MiDaS_small"  # Options: "MiDaS_small", "DPT_Large", "DPT_Hybrid"

# Sea-Thru algorithm parameters
# Attenuation coefficients for underwater light (RGB channels)
# These are approximate values and can be tuned based on water conditions
# Reduced values to prevent over-correction
BETA_COEFFICIENTS = {
    'blue': 0.3,    # Reduced from 0.5 - Blue channel
    'green': 0.15,  # Reduced from 0.2 - Green channel  
    'red': 0.08     # Reduced from 0.1 - Red channel
}

# Backscatter estimation parameters
BACKSCATTER_FRACTION = 0.01  # Fraction of pixels to use for backscatter estimation
DARK_PIXEL_FRACTION = 0.1    # Fraction of darkest pixels from farthest region

# Image processing parameters
ENHANCE_CONTRAST = True
ENHANCE_SATURATION = True
SATURATION_BOOST = 1.25  # Natural vibrancy matching reference (f)
CONTRAST_ALPHA = 1.1     # Moderate contrast enhancement

# HD Quality Enhancement
ENABLE_SUPER_RESOLUTION = False  # Enable 2x upscaling for 4K output (slower)
ADVANCED_DENOISING = True        # Use non-local means denoising (better quality)
DETAIL_ENHANCEMENT = True        # Enhance fine details and textures
EDGE_SHARPENING = True          # Edge-aware sharpening for clarity

# Server settings
HOST = "0.0.0.0"
PORT = 8000
RELOAD = True  # Set to False in production
