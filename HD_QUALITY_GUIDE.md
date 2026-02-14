# HD Quality Enhancement Configuration Guide

This guide explains the advanced quality enhancement options in `config.py`.

## Quality Enhancement Options

### 1. ADVANCED_DENOISING (Default: True)
**What it does**: Uses Non-Local Means Denoising instead of bilateral filtering
- **Pros**: Superior noise reduction, preserves fine details better
- **Cons**: Slower processing (~2-3x longer)
- **Recommended**: True for best quality

### 2. DETAIL_ENHANCEMENT (Default: True)
**What it does**: Enhances fine textures and details using high-pass filtering
- **Effect**: Makes coral textures, fish scales, and small details more visible
- **Recommended**: True for HD quality

### 3. EDGE_SHARPENING (Default: True)
**What it does**: Applies strong sharpening only to edges (edge-aware)
- **Effect**: Crystal-clear edges without artifacts in smooth areas
- **Recommended**: True for professional results

### 4. ENABLE_SUPER_RESOLUTION (Default: False)
**What it does**: 2x upscaling for 4K output
- **Input**: 1080p (1920x1080) → **Output**: 4K (3840x2160)
- **Pros**: Higher resolution output
- **Cons**: 4-5x slower, larger file sizes
- **Recommended**: 
  - `True` if you want 4K output
  - `False` for faster processing

## Performance Impact

| Mode | Processing Time | Quality |
|------|----------------|---------|
| Fast (all False) | ~5 seconds | Good |
| **Standard** (defaults) | ~15 seconds | **Excellent** |
| 4K (super-res enabled) | ~60 seconds | Ultra HD |

## How to Enable 4K Output

Edit `d:\projects\robust_underwater_app\backend\config.py`:

```python
# Change this line from False to True
ENABLE_SUPER_RESOLUTION = True
```

Then restart the server.

## Recommended Settings

**For Best Quality + Speed:**
```python
ADVANCED_DENOISING = True
DETAIL_ENHANCEMENT = True  
EDGE_SHARPENING = True
ENABLE_SUPER_RESOLUTION = False  # Keep False unless you need 4K
```

**For Maximum Quality (4K):**
```python
ADVANCED_DENOISING = True
DETAIL_ENHANCEMENT = True
EDGE_SHARPENING = True
ENABLE_SUPER_RESOLUTION = True  # Enable 4K
```

**For Fast Preview:**
```python
ADVANCED_DENOISING = False
DETAIL_ENHANCEMENT = False
EDGE_SHARPENING = False
ENABLE_SUPER_RESOLUTION = False
```
