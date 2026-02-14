# Batch Testing Guide

This guide explains how to test the underwater enhancement algorithm on multiple images.

## Important Note

⚠️ **This is NOT a machine learning model** - it's a traditional computer vision algorithm with fixed parameters. We cannot "train" it, but we can:
- **Test** on multiple images
- **Tune parameters** based on results
- **Find optimal settings** for your specific images

---

## Option 1: Batch Process Many Images

Process all images in a folder at once.

### Steps:

1. **Prepare your images:**
   - Put all underwater images in one folder
   - Example: `D:\my_underwater_images\`

2. **Run the batch processor:**
   ```bash
   python batch_process.py
   ```

3. **Choose option 1**

4. **Enter paths:**
   - Input folder: `D:\my_underwater_images\`
   - Output folder: `D:\enhanced_output\` (or press Enter for default)

5. **Wait for processing** - all images will be enhanced

6. **Review results** in the output folder

---

## Option 2: Compare Parameters on One Image

Test different parameter settings on a single image to find what works best.

### Steps:

1. **Choose a representative test image**
   - Pick one that represents your typical underwater photo

2. **Run the batch processor:**
   ```bash
   python batch_process.py
   ```

3. **Choose option 2**

4. **Enter image path:**
   - Example: `D:\my_underwater_images\test.jpg`

5. **View results:**
   - Opens comparison with 4 different parameter sets:
     - **Conservative** - minimal changes
     - **Balanced** - current default (like reference f)
     - **Vibrant** - stronger colors
     - **Aggressive** - maximum correction

6. **Check `comparison_grid.jpg`** to see all versions side-by-side

---

## Option 3: Manual Parameter Tuning

If you want specific parameter control:

### Edit `backend/config.py`:

```python
# Adjust these values based on your test results:

SATURATION_BOOST = 1.25  # Range: 1.0-2.0
                         # 1.0 = no boost
                         # 1.25 = balanced (current)
                         # 1.5+ = very vibrant

ENHANCE_CONTRAST = True   # Set False to disable
ENHANCE_SATURATION = True # Set False to disable
```

### In `backend/sea_thru.py`, line 182-188:

```python
# Red compensation (line 182)
red_compensated = red_channel + (green_channel - red_channel) * depth_factor * 0.45
# Increase 0.45 → 0.6 for warmer tones
# Decrease 0.45 → 0.3 for cooler tones

# Blue reduction (line 186)
blue_reduced = blue_channel * (1.0 - depth_factor * 0.3)
# Increase 0.3 → 0.5 to remove more blue
# Decrease 0.3 → 0.2 to keep more blue

# Green reduction (line 189)
green_adjusted = green_channel * (1.0 - depth_factor * 0.25)
# Increase 0.25 → 0.4 to remove more green
# Decrease 0.25 → 0.15 to keep more green
```

---

## Recommended Workflow

1. **Start with batch processing** (Option 1) to see how current settings work on all your images

2. **If results are inconsistent:**
   - Use parameter comparison (Option 2) on 2-3 representative images
   - Identify which parameter set works best

3. **Fine-tune manually** if needed:
   - Edit config.py with values from the best parameter set
   - Re-run batch processing

4. **Test final settings:**
   - Process all images again
   - Verify results are consistent

---

## Troubleshooting

### Images too warm/red:
- Decrease `red_factor` (line 182 in sea_thru.py)
- Reduce `SATURATION_BOOST` in config.py

### Images still too blue:
- Increase `blue_reduction` (line 186)
- Increase `green_reduction` (line 189)

### Images too dark:
- Increase `gamma` (line 224 in sea_thru.py) from 1.05 to 1.1-1.15

### Colors too flat:
- Increase `SATURATION_BOOST` in config.py
- Increase `CLAHE clipLimit` (line 233) from 2.5 to 3.0

---

## Example Commands

```bash
# Process all images in a folder
python batch_process.py
# Choose 1, then enter: D:\underwater_photos\

# Compare parameters on one image  
python batch_process.py
# Choose 2, then enter: D:\underwater_photos\sample.jpg

# Use web interface for single images
python main.py
# Go to http://localhost:8000
```
