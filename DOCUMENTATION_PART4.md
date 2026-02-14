# User Guide, Developer Guide, Troubleshooting & References

# 12. User Guide

## 12.1 Getting Started

### 12.1.1 First-Time Setup

**Step 1: Installation**

See [Installation Guide](README.md#installation) for detailed setup instructions.

**Step 2: Starting the Application**

```bash
# Windows
cd d:\projects\robust_underwater_app
venv\Scripts\activate
python main.py

# Linux/Mac
cd /path/to/robust_underwater_app
source venv/bin/activate
python main.py
```

**Step 3: Access Web Interface**

Open your browser and navigate to:
```
http://localhost:8000
```

### 12.1.2 Processing Your First Image

1. **Select Image**: Click "Choose Image" or drag-and-drop
2. **Wait for Upload**: File name will appear
3. **Process**: Click "Process Image" button
4. **View Results**: Compare original vs. enhanced
5. **Download**: Click download button to save results

## 12.2 Best Practices

### 12.2.1 Image Selection

**Good Input Images:**
- Clear underwater scenes
- Visible subjects (fish, coral, divers)
- Various depths in scene
- RAW or high-quality JPEG

**Challenging Images:**
- Extremely murky water
- Total darkness (no ambient light)
- Pure white/black subjects
- Heavy motion blur

### 12.2.2 Quality Tips

**For Best Results:**

1. **Use Original Resolution**: Don't downscale before processing
2. **Avoid Over-Processed Images**: Process RAW when possible
3. **Multiple Attempts**: Try different parameter settings
4. **Batch Similar Images**: Process groups with similar conditions

**Common Issues:**

| Issue | Solution |
|-------|----------|
| Too blue/green | Increase beta coefficients |
| Too warm/red | Decrease saturation boost |
| Noisy output | Enable advanced denoising |
| Too dark | Increase gamma correction |
| Washed out | Increase CLAHE clip limit |

## 12.3 Advanced Features

### 12.3.1 Batch Processing

Process multiple images at once:

```bash
# Using batch processor
python batch_process.py

# Choose option 1
# Enter folder path: D:\underwater_photos
# Enter output path: D:\enhanced_photos

# Wait for completion
```

### 12.3.2 Parameter Comparison

Test different settings on one image:

```bash
python batch_process.py

# Choose option 2
# Enter image path: D:\test_image.jpg
# Output: Creates comparison grid with 4 parameter sets
```

### 12.3.3 HD/4K Output

Enable super-resolution in `backend/config.py`:

```python
ENABLE_SUPER_RESOLUTION = True  # 2x upscaling
```

**Note**: This significantly increases processing time (5-10x slower).

## 12.4 Troubleshooting

### 12.4.1 Common Errors

**Error: "No module named 'cv2'"**
```bash
Solution:
pip install opencv-python
```

**Error: "Cannot load MiDaS model"**
```bash
Solution:
1. Check internet connection
2. Clear cache: rm -rf ~/.cache/torch/hub
3. Reinstall torch: pip install --upgrade torch
```

**Error: "File too large"**
```python
Solution:
Edit config.py:
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
```

**Error: "Processing timeout"**
```python
Solution:
Use smaller model:
MIDAS_MODEL = "MiDaS_small"
```

### 12.4.2 Performance Issues

**Slow Processing:**

1. **Use Smaller Model**:
   ```python
   MIDAS_MODEL = "MiDaS_small"  # Fastest
   ```

2. **Disable Expensive Features**:
   ```python
   ENABLE_SUPER_RESOLUTION = False
   ADVANCED_DENOISING = False
   ```

3. **Reduce Image Size**: Downscale before processing
   ```python
   img = cv2.resize(img, None, fx=0.5, fy=0.5)
   ```

4. **Use GPU**: If available
   ```python
   # Automatically used if CUDA available
   # Check: torch.cuda.is_available()
   ```

---

# 13. Developer Guide

## 13.1 Architecture Overview

### 13.1.1 Module Organization

```
backend/
├── main.py              # FastAPI app and endpoints
├── sea_thru.py          # Core algorithm
├── depth_estimation.py  # MiDaS integration
└── config.py            # Configuration

frontend/
├── index.html           # UI structure
├── style.css            # Styling
└── app.js               # Client logic
```

### 13.1.2 Key Design Patterns

**1. Separation of Concerns**
- Depth estimation separate from color recovery
- Configuration centralized
- API layer separate from processing

**2. Pipeline Pattern**
- Sequential processing stages
- Each stage outputs input for next

**3. Dependency Injection**
- DepthEstimator initialized once
- Reused across requests

## 13.2 Adding New Features

### 13.2.1 Adding New Enhancement Step

**Example: Add histogram equalization**

```python
# In sea_thru.py

def histogram_equalization(img):
    """Apply histogram equalization to improve contrast"""
    # Convert to YUV
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    
    # Equalize Y channel
    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
    
    # Convert back
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

def recover_colors(img, depth, B_inf, illuminant, beta):
    # ... existing code ...
    
    # Add histogram equalization step
    if config.ENABLE_HISTOGRAM_EQ:
        enhanced_uint8 = (enhanced * 255).astype(np.uint8)
        enhanced_uint8 = histogram_equalization(enhanced_uint8)
        enhanced = enhanced_uint8.astype(np.float32) / 255.0
    
    # ... rest of code ...
```

**Add configuration:**

```python
# In config.py
ENABLE_HISTOGRAM_EQ = False
```

### 13.2.2 Adding New API Endpoint

**Example: Add batch processing endpoint**

```python
# In main.py

from typing import List

@app.post("/batch_process")
async def batch_process(files: List[UploadFile] = File(...)):
    """Process multiple images at once"""
    results = []
    
    for file in files:
        # Validate
        if not file:
            continue
        
        # Save
        upload_path = os.path.join(config.UPLOAD_DIR, file.filename)
        with open(upload_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        try:
            # Process
            depth = depth_estimator.estimate(upload_path)
            img = cv2.imread(upload_path)
            enhanced = sea_thru_pipeline(img, depth)
            
            # Save
            output_filename = f"enhanced_{file.filename}"
            output_path = os.path.join(config.OUTPUT_DIR, output_filename)
            cv2.imwrite(output_path, enhanced)
            
            results.append({
                "filename": file.filename,
                "success": True,
                "enhanced": f"/outputs/{output_filename}"
            })
        
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return JSONResponse({"results": results})
```

### 13.2.3 Custom Depth Estimator

**Example: Add alternative depth model**

```python
# In depth_estimation.py

class CustomDepthEstimator:
    def __init__(self):
        # Load your custom model
        self.model = load_custom_model()
    
    def estimate(self, img_path):
        img = cv2.imread(img_path)
        # Your custom depth estimation
        depth = self.model.predict(img)
        return depth

# In main.py
if config.DEPTH_MODEL == "custom":
    from depth_estimation import CustomDepthEstimator
    depth_estimator = CustomDepthEstimator()
else:
    depth_estimator = DepthEstimator()
```

## 13.3 Code Style Guidelines

### 13.3.1 Python Code Style

Follow PEP 8:

```python
# Good
def estimate_backscatter(img, fraction=0.001):
    """
    Estimate backscatter using dark pixels.
    
    Args:
        img: Input image (BGR)
        fraction: Fraction of darkest pixels
    
    Returns:
        backscatter: BGR array
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ... implementation ...
    return backscatter

# Bad
def EstimateBackscatter(img,fraction=0.001):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return backscatter
```

**Key Points:**
- Use snake_case for functions/variables
- Use CamelCase for classes
- 4 spaces for indentation
- Docstrings for all functions
- Type hints where appropriate

### 13.3.2 JavaScript Code Style

```javascript
// Good
async function processImage(file) {
    try {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch('/process', {
            method: 'POST',
            body: formData
        });
        
        return await response.json();
    } catch (error) {
        console.error('Processing failed:', error);
        throw error;
    }
}

// Bad  
async function ProcessImage(file){
try{
const formData=new FormData()
formData.append('file',file)
const response=await fetch('/process',{method:'POST',body:formData})
return await response.json()}catch(error){console.error(error);throw error}}
```

## 13.4 Contributing

### 13.4.1 Development Workflow

**1. Fork Repository**

**2. Create Feature Branch**
```bash
git checkout -b feature/new-enhancement
```

**3. Make Changes**
- Write code
- Add tests
- Update documentation

**4. Test**
```bash
pytest tests/ -v
```

**5. Commit**
```bash
git add .
git commit -m "Add new enhancement feature"
```

**6. Push**
```bash
git push origin feature/new-enhancement
```

**7. Create Pull Request**

### 13.4.2 Code Review Checklist

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
- [ ] Performance impact considered
- [ ] Error handling added

---

# 14. Performance Optimization

## 14.1 Optimization Strategies

### 14.1.1 Algorithm Optimization

**1. Use NumPy Vectorization**

```python
# Slow (loops)
for i in range(height):
    for j in range(width):
        enhanced[i, j] = img[i, j] * factor

# Fast (vectorized)
enhanced = img * factor
```

**2. Reduce Unnecessary Copies**

```python
# Slow (creates copies)
img_copy1 = img.copy()
img_copy2 = img_copy1.copy()

# Fast (in-place or reuse)
output = img  # No copy
cv2.someFunction(output, output)  # In-place
```

**3. Use Efficient Data Types**

```python
# Memory-efficient
img_float = img.astype(np.float32)  # 32-bit float

# Memory-inefficient  
img_double = img.astype(np.float64)  # 64-bit float
```

### 14.1.2 Model Optimization

**1. Model Quantization**

```python
# Quantize MiDaS to INT8 (faster inference)
import torch.quantization

model_fp32 = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32,
    {torch.nn.Linear},
    dtype=torch.qint8
)
```

**2. TorchScript Compilation**

```python
# Compile model for faster execution
scripted_model = torch.jit.script(model)
scripted_model.save("midas_scripted.pt")

# Load compiled model
model = torch.jit.load("midas_scripted.pt")
```

**3. ONNX Export**

```python
# Export to ONNX for broader deployment
dummy_input = torch.randn(1, 3, 384, 384)
torch.onnx.export(
    model,
    dummy_input,
    "midas.onnx",
    opset_version=11
)

# Use with ONNX Runtime (faster CPU inference)
import onnxruntime as ort
session = ort.InferenceSession("midas.onnx")
```

### 14.1.3 Caching Strategies

**1. Depth Map Caching**

```python
# Cache depth maps to avoid recomputation
depth_cache = {}

def get_depth_cached(img_path):
    if img_path in depth_cache:
        return depth_cache[img_path]
    
    depth = depth_estimator.estimate(img_path)
    depth_cache[img_path] = depth
    return depth
```

**2. Model Caching**

```python
# Load model once, reuse across requests
@lru_cache(maxsize=1)
def get_depth_estimator():
    return DepthEstimator()

# Use singleton
depth_estimator = get_depth_estimator()
```

## 14.2 Profiling and Benchmarking

### 14.2.1 Python Profiling

```python
# Using cProfile
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Code to profile
enhanced = sea_thru_pipeline(img, depth)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

### 14.2.2 Line Profiling

```python
# Using line_profiler
# pip install line_profiler

# Add @profile decorator
@profile
def recover_colors(img, depth, B_inf, illuminant, beta):
    # ... code ...

# Run with:
# kernprof -l -v script.py
```

### 14.2.3 Memory Profiling

```python
# Using memory_profiler
# pip install memory_profiler

from memory_profiler import profile

@profile
def sea_thru_pipeline(img, depth):
    # ... code ...

# Run with:
# python -m memory_profiler script.py
```

---

# 15. Research Background

## 15.1 Sea-Thru Paper Summary

**Title**: Sea-Thru: A Method for Removing Water from Underwater Images  
**Authors**: Derya Akkaynak, Tali Treibitz  
**Published**: CVPR 2019

### Key Contributions

1. **Physically Accurate Model**: Based on actual light propagation in water
2. **Range Dependency**: Accounts for distance-dependent color loss
3. **Backscatter Estimation**: Novel method for estimating veiling light
4. **Illuminant Recovery**: Estimates spatially-varying illumination

### Mathematical Model

The underwater image formation:

```
I_c(x) = J_c(x) · e^(-β_c·d(x)) + B_c^∞ · (1 - e^(-β_c·d(x)))
```

Where:
- `c ∈ {R,G,B}`: Color channel
- `I_c`: Observed intensity
- `J_c`: True scene radiance (to recover)
- `β_c`: Attenuation coefficient
- `d(x)`: Distance to point x
- `B_c^∞`: Backscatter at infinity

### Recovery Process

1. **Estimate Range**: Use stereo or depth sensor
2. **Estimate Backscatter**: From dark pixels
3. **Estimate Beta**: From image statistics
4. **Invert Model**: Solve for J_c(x)

## 15.2 MiDaS Paper Summary

**Title**: Towards Robust Monocular Depth Estimation  
**Authors**: Ranftl et al.  
**Published**: CVPR 2020 (v1), PAMI 2021 (v3)

### Key Features

1. **Mixed Dataset Training**: Multiple depth datasets
2. **Relative Depth**: Outputs disparity, not metric depth
3. **Robust**: Works across diverse scenes
4. **Efficient**: Multiple model sizes

### Architecture

- **Encoder**: Vision Transformer or EfficientNet
- **Decoder**: Multi-scale feature fusion
- **Output**: Dense depth map at input resolution

---

# 16. Appendices

## Appendix A: Mathematical Derivations

### A.1 underwater Image Formation

Starting from the radiative transfer equation:

```
dL(z)/dz = -β·L(z) + β·B^∞
```

Solving this ODE:

```
L(z) = L(0)·e^(-β·z) + B^∞·(1 - e^(-β·z))
```

This is the underwater image formation model used in Sea-Thru.

### A.2 Color Recovery Derivation

Given:
```
I_c = J_c · t_c + B_c^∞ · (1 - t_c)
```

Where `t_c = e^(-β_c·d)`, solve for `J_c`:

```
I_c = J_c · t_c + B_c^∞ - B_c^∞ · t_c
I_c - B_c^∞ = J_c · t_c - B_c^∞ · t_c
I_c - B_c^∞ = t_c · (J_c - B_c^∞)
J_c = (I_c - B_c^∞) / t_c + B_c^∞
```

## Appendix B: Configuration Reference

### B.1 Complete Config File

```python
# config.py - Full reference

import os

# =============== File Upload Settings ===============
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB in bytes
ALLOWED_EXTENSIONS = {
    '.jpg', '.jpeg',  # JPEG images
    '.png',           # PNG images
    '.bmp',           # Bitmap images
    '.tiff', '.tif'   # TIFF images
}

# =============== Directory Settings ===============
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
STATIC_DIR = "frontend"

# Auto-create directories
for directory in [UPLOAD_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

# =============== MiDaS Settings ===============
MIDAS_MODEL = "MiDaS_small"
# Options:
#   - "MiDaS_small": Fastest, good quality
#   - "DPT_Hybrid": Medium speed, better quality
#   - "DPT_Large": Slowest, best quality

# =============== Sea-Thru Parameters ===============
BETA_COEFFICIENTS = {
    'blue': 0.3,    # Blue attenuation (0.1-0.5)
    'green': 0.15,  # Green attenuation (0.05-0.3)
    'red': 0.08     # Red attenuation (0.03-0.15)
}

BACKSCATTER_ESTIMATION_METHOD = "dark_pixels"
DARK_PIXEL_FRACTION = 0.001  # Use darkest 0.1% of pixels

# =============== Enhancement Parameters ===============
ENHANCE_CONTRAST = True
ENHANCE_SATURATION = True

SATURATION_BOOST = 1.25  # Range: 1.0-2.0
CONTRAST_ALPHA = 1.1     # Range: 1.0-1.5

# =============== HD Quality Settings ===============
ADVANCED_DENOISING = True      # Non-Local Means (slow but effective)
DETAIL_ENHANCEMENT = True       # High-pass sharpening
EDGE_SHARPENING = True         # Unsharp mask
ENABLE_SUPER_RESOLUTION = False  # 2x upscaling (very slow!)

# Denoising parameters
NLM_H = 10           # Luminance denoising strength
NLM_H_COLOR = 10     # Color denoising strength
NLM_TEMPLATE = 7     # Template window size
NLM_SEARCH = 21      # Search window size

# CLAHE parameters
CLAHE_CLIP_LIMIT = 2.5  # Contrast limiting (1.0-4.0)
CLAHE_TILE_SIZE = 8     # Grid size for local equalization

# =============== Server Settings ===============
SERVER_HOST = "0.0.0.0"  # Bind to all interfaces
SERVER_PORT = 8000        # HTTP port
RELOAD = True             # Auto-reload on file changes (dev only)
DEBUG = False             # Debug mode

# =============== Performance Settings ===============
MAX_WORKERS = 4  # For concurrent processing
TIMEOUT = 300    # Request timeout in seconds
```

## Appendix C: Glossary

| Term | Definition |
|------|------------|
| **Attenuation** | Reduction of light intensity as it travels through water |
| **Backscatter** | Light scattered back toward the camera by water particles |
| **CLAHE** | Contrast Limited Adaptive Histogram Equalization |
| **Depth Map** | 2D representation of distance to each pixel |
| **Disparity** | Inverse depth (closer objects have larger values) |
| **MiDaS** | Monocular Depth Estimation model |
| **Sea-Thru** | Algorithm for underwater color restoration |
| **Transmission** | Fraction of light that reaches the camera |

## Appendix D: References

1. Akkaynak, D., & Treibitz, T. (2019). Sea-Thru: A Method for Removing Water From Underwater Images. CVPR 2019.

2. Ranftl, R., Bochkovskiy, A., & Koltun, V. (2021). Vision Transformers for Dense Prediction. ICCV 2021.

3. Ranftl, R., Lasinger, K., Hafner, D., Schindler, K., & Koltun, V. (2020). Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer. TPAMI.

4. Ancuti, C., Ancuti, C. O., Haber, T., & Bekaert, P. (2012). Enhancing Underwater Images and Videos by Fusion. CVPR 2012.

5. Drews, P., Nascimento, E., Moraes, F., Botelho, S., & Campos, M. (2013). Transmission Estimation in Underwater Single Images. ICCV Workshops 2013.

6. He, K., Sun, J., & Tang, X. (2011). Single Image  Haze Removal Using Dark Channel Prior. TPAMI.

## Appendix E: License

**MIT License**

Copyright (c) 2026 Underwater Enhancement Project

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

**END OF DOCUMENTATION**

*Total Pages: 120+*  
*Version: 1.0*  
*Last Updated: February 2026*
