# Implementation Details (Continued)

## 6.2 Code Walkthrough

### 6.2.1 Main Application Flow

```python
# main.py - Complete flow

@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    """
    Process uploaded underwater image
    
    Flow:
    1. Validate file type and size
    2. Save to uploads directory
    3. Estimate depth using MiDaS
    4. Apply Sea-Thru pipeline
    5. Save enhanced image
    6. Return file paths
    """
    
    # Step 1: Validation
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {config.ALLOWED_EXTENSIONS}"
        )
    
    # Step 2: Save upload
    upload_path = os.path.join(config.UPLOAD_DIR, file.filename)
    with open(upload_path, "wb") as buffer:
        content = await file.read()
        if len(content) > config.MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large")
        buffer.write(content)
    
    try:
        # Step 3: Depth estimation
        depth_map = depth_estimator.estimate(upload_path)
        
        # Step 4: Read image for processing
        img = cv2.imread(upload_path)
        
        # Step 5: Apply Sea-Thru
        enhanced = sea_thru_pipeline(img, depth_map)
        
        # Step 6: Save result
        output_filename = f"enhanced_{file.filename}"
        output_path = os.path.join(config.OUTPUT_DIR, output_filename)
        cv2.imwrite(output_path, enhanced)
        
        # Step 7: Return response
        return JSONResponse({
            "success": True,
            "original": f"/uploads/{file.filename}",
            "enhanced": f"/outputs/{output_filename}",
            "filename": output_filename
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 6.2.2 Depth Estimation Details

```python
# depth_estimation.py

class DepthEstimator:
    def __init__(self, model_type="MiDaS_small"):
        """
        Initialize MiDaS depth estimator
        
        Args:
            model_type: "MiDaS_small", "DPT_Large", or "DPT_Hybrid"
        """
        # Device selection (GPU if available)
        self.device = torch.device("cuda") if torch.cuda.is_available() \
                      else torch.device("cpu")
        print(f"Using device: {self.device}")
        
        # Load model from torch hub
        print(f"Loading MiDaS model: {model_type}...")
        self.model = torch.hub.load(
            "intel-isl/MiDaS",
            model_type,
            trust_repo=True
        )
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Load appropriate transforms
        midas_transforms = torch.hub.load(
            "intel-isl/MiDaS",
            "transforms",
            trust_repo=True
        )
        
        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform
        
        print("MiDaS model loaded successfully!")
    
    def estimate(self, img_path):
        """
        Estimate depth map from image
        
        Args:
            img_path: Path to input image
        
        Returns:
            depth_map: Normalized depth map [0, 1]
                      where 0 = close, 1 = far
        """
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        input_batch = self.transform(img_rgb).to(self.device)
        
        # Inference
        with torch.no_grad():
            prediction = self.model(input_batch)
            
            # Resize to original resolution
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        # Convert to numpy
        depth_map = prediction.cpu().numpy()
        
        # MiDaS outputs inverse depth (disparity)
        # Invert to get actual depth
        depth_map = depth_map.max() - depth_map
        
        # Normalize to [0, 1]
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        if depth_max - depth_min > 1e-6:
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        else:
            depth_map = np.ones_like(depth_map) * 0.5
        
        return depth_map
```

### 6.2.3 Sea-Thru Pipeline Implementation

```python
# sea_thru.py

def estimate_backscatter_from_dark_pixels(img, fraction=0.001):
    """
    Estimate backscatter using dark channel prior
    
    Args:
        img: Input image (BGR)
        fraction: Fraction of darkest pixels to use
    
    Returns:
        backscatter: BGR backscatter estimate
    """
    # Convert to grayscale for dark pixel detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Flatten and find darkest pixels
    flat_gray = gray.flatten()
    num_dark_pixels = int(len(flat_gray) * fraction)
    
    # Get indices of darkest pixels
    dark_indices = np.argpartition(flat_gray, num_dark_pixels)[:num_dark_pixels]
    
    # Get corresponding color values
    flat_img = img.reshape(-1, 3)
    dark_pixels = flat_img[dark_indices]
    
    # Average to get backscatter estimate
    backscatter = dark_pixels.mean(axis=0)
    
    return backscatter

def recover_colors(img, depth, B_inf, illuminant, beta):
    """
    Recover true scene colors using depth-adaptive enhancement
    
    Args:
        img: BGR image (uint8)
        depth: Normalized depth map [0=close, 1=far]
        B_inf: Backscatter estimate
        illuminant: Illuminant normalization
        beta: Attenuation coefficients (BGR)
    
    Returns:
        recovered: Enhanced image (BGR, uint8)
    """
    # Convert to float [0, 1]
    img_float = img.astype(np.float32) / 255.0
    original_shape = img.shape[:2]
    
    # Optional: Super-resolution upscaling
    if config.ENABLE_SUPER_RESOLUTION:
        img_float = cv2.resize(
            img_float, None,
            fx=2, fy=2,
            interpolation=cv2.INTER_CUBIC
        )
        depth = cv2.resize(
            depth,
            (img_float.shape[1], img_float.shape[0]),
            interpolation=cv2.INTER_CUBIC
        )
    
    # Channel separation
    blue_channel = img_float[:, :, 0]
    green_channel = img_float[:, :, 1]
    red_channel = img_float[:, :, 2]
    
    # Depth factor for adaptive processing
    depth_factor = np.clip(depth, 0, 1)
    
    # Red channel compensation
    # Red is most absorbed, boost using green as reference
    red_compensated = red_channel + \
                      (green_channel - red_channel) * depth_factor * 0.45
    red_compensated = np.clip(red_compensated, 0, 1)
    
    # Blue reduction (remove blue cast)
    blue_reduced = blue_channel * (1.0 - depth_factor * 0.3)
    
    # Green reduction (remove cyan tint)
    green_adjusted = green_channel * (1.0 - depth_factor * 0.25)
    
    # Reconstruct image
    enhanced = np.stack([blue_reduced, green_adjusted, red_compensated], axis=2)
    enhanced = np.clip(enhanced, 0, 1)
    
    # White balance
    mean_b, mean_g, mean_r = enhanced.mean(axis=(0, 1))
    avg_intensity = (mean_b + mean_g + mean_r) / 3.0
    
    if avg_intensity > 0:
        scale_b = avg_intensity / (mean_b + 1e-6)
        scale_g = avg_intensity / (mean_g + 1e-6)
        scale_r = avg_intensity / (mean_r + 1e-6)
        
        blend = 0.4  # Conservative blending
        enhanced[:, :, 0] = np.clip(
            enhanced[:, :, 0] * (scale_b * blend + (1 - blend)), 0, 1
        )
        enhanced[:, :, 1] = np.clip(
            enhanced[:, :, 1] * (scale_g * blend + (1 - blend)), 0, 1
        )
        enhanced[:, :, 2] = np.clip(
            enhanced[:, :, 2] * (scale_r * blend + (1 - blend)), 0, 1
        )
    
    # Advanced denoising
    if config.ADVANCED_DENOISING:
        enhanced_uint8 = (enhanced * 255).astype(np.uint8)
        denoised = cv2.fastNlMeansDenoisingColored(
            enhanced_uint8, None,
            h=10, hColor=10,
            templateWindowSize=7,
            searchWindowSize=21
        )
        enhanced = denoised.astype(np.float32) / 255.0
    
    # Gamma correction for brightness
    gamma = 1.05
    enhanced = np.power(enhanced, gamma)
    
    # Contrast enhancement (CLAHE)
    if config.ENHANCE_CONTRAST:
        enhanced_uint8 = (enhanced * 255).astype(np.uint8)
        lab = cv2.cvtColor(enhanced_uint8, cv2.COLOR_BGR2LAB)
        
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        enhanced_uint8 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        enhanced = enhanced_uint8.astype(np.float32) / 255.0
    
    # Detail enhancement
    if config.DETAIL_ENHANCEMENT:
        enhanced_uint8 = (enhanced * 255).astype(np.uint8)
        
        # High-pass filter for detail
        gaussian = cv2.GaussianBlur(enhanced_uint8, (0, 0), 2.0)
        detail = cv2.addWeighted(enhanced_uint8, 1.5, gaussian, -0.5, 0)
        
        enhanced = detail.astype(np.float32) / 255.0
    
    # Edge sharpening
    if config.EDGE_SHARPENING:
        enhanced_uint8 = (enhanced * 255).astype(np.uint8)
        
        # Unsharp mask
        gaussian_blur = cv2.GaussianBlur(enhanced_uint8, (5, 5), 1.0)
        sharpened = cv2.addWeighted(enhanced_uint8, 1.5, gaussian_blur, -0.5, 0)
        
        enhanced = sharpened.astype(np.float32) / 255.0
    
    # Saturation boost
    if config.ENHANCE_SATURATION:
        enhanced_uint8 = (enhanced * 255).astype(np.uint8)
        hsv = cv2.cvtColor(enhanced_uint8, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        hsv[:, :, 1] = np.clip(
            hsv[:, :, 1] * config.SATURATION_BOOST, 0, 255
        )
        
        enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        enhanced = enhanced.astype(np.float32) / 255.0
    
    # Final conversion
    recovered = (enhanced * 255).astype(np.uint8)
    
    return recovered

def sea_thru_pipeline(img, depth):
    """
    Complete Sea-Thru processing pipeline
    
    Args:
        img: Input image (BGR, uint8)
        depth: Depth map [0, 1]
    
    Returns:
        Enhanced image (BGR, uint8)
    """
    # Estimate backscatter
    B_inf = estimate_backscatter_from_dark_pixels(img)
    
    # Illuminant (not actively used)
    illuminant = np.array([255, 255, 255])
    
    # Beta coefficients
    beta = np.array([
        config.BETA_COEFFICIENTS['blue'],
        config.BETA_COEFFICIENTS['green'],
        config.BETA_COEFFICIENTS['red']
    ])
    
    # Recover colors
    recovered = recover_colors(img, depth, B_inf, illuminant, beta)
    
    return recovered
```

---

# 7. Frontend Development

## 7.1 HTML Structure

### 7.1.1 Page Layout

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Underwater Image Enhancement</title>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header>
            <h1>🌊 Underwater Image Enhancement</h1>
            <p>Restore true colors from underwater photos</p>
        </header>
        
        <!-- Upload Section -->
        <section class="upload-section">
            <div class="upload-box">
                <input type="file" id="imageInput" accept="image/*">
                <label for="imageInput" class="upload-label">
                    <span>📁 Choose Image</span>
                    <span class="file-info">or drag and drop</span>
                </label>
            </div>
            <button id="processBtn" class="process-btn" disabled>
                Process Image
            </button>
        </section>
        
        <!-- Progress Section -->
        <section id="progressSection" class="progress-section" style="display: none;">
            <div class="progress-bar">
                <div class="progress-fill" id="progressFill"></div>
            </div>
            <p id="progressText" class="progress-text">Processing...</p>
        </section>
        
        <!-- Results Section -->
        <section id="resultsSection" class="results-section" style="display: none;">
            <div class="image-comparison">
                <div class="image-container">
                    <h3>Original</h3>
                    <img id="originalImage" alt="Original">
                    <button class="download-btn" onclick="downloadImage('original')">
                        ⬇️ Download
                    </button>
                </div>
                <div class="image-container">
                    <h3>Enhanced</h3>
                    <img id="enhancedImage" alt="Enhanced">
                    <button class="download-btn" onclick="downloadImage('enhanced')">
                        ⬇️ Download
                    </button>
                </div>
            </div>
            <button id="newImageBtn" class="new-image-btn">
                Process Another Image
            </button>
        </section>
    </div>
    
    <script src="/static/app.js"></script>
</body>
</html>
```

## 7.2 CSS Styling

### 7.2.1 Glassmorphism Design

```css
:root {
    --primary-color: #00a8e8;
    --secondary-color: #007ea7;
    --accent-color: #00bcd4;
    --bg-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --glass-bg: rgba(255, 255, 255, 0.1);
    --glass-border: rgba(255, 255, 255, 0.2);
}

body {
    margin: 0;
    padding: 0;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: var(--bg-gradient);
    min-height: 100vh;
    display: flex;
    justify-content: center;
    align-items: center;
}

.container {
    width: 90%;
    max-width: 1200px;
    background: var(--glass-bg);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    border: 1px solid var(--glass-border);
    padding: 40px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.37);
}

header {
    text-align: center;
    color: white;
    margin-bottom: 30px;
}

header h1 {
    font-size: 2.5em;
    margin: 0 0 10px 0;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
}

header p {
    font-size: 1.2em;
    opacity: 0.9;
    margin: 0;
}
```

### 7.2.2 Upload Section Styling

```css
.upload-section {
    text-align: center;
    margin: 30px 0;
}

.upload-box {
    position: relative;
    margin-bottom: 20px;
}

.upload-box input[type="file"] {
    display: none;
}

.upload-label {
    display: inline-block;
    padding: 40px 80px;
    background: var(--glass-bg);
    border: 2px dashed var(--glass-border);
    border-radius: 15px;
    cursor: pointer;
    transition: all 0.3s ease;
}

.upload-label:hover {
    background: rgba(255, 255, 255, 0.2);
    border-color: var(--accent-color);
    transform: translateY(-2px);
}

.upload-label span {
    display: block;
    color: white;
    font-size: 1.2em;
}

.file-info {
    font-size: 0.9em !important;
    opacity: 0.7;
    margin-top: 10px;
}

.process-btn {
    padding: 15px 40px;
    font-size: 1.1em;
    background: var(--accent-color);
    color: white;
    border: none;
    border-radius: 10px;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0, 188, 212, 0.4);
}

.process-btn:hover:not(:disabled) {
    background: var(--primary-color);
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0, 188, 212, 0.6);
}

.process-btn:disabled {
    background: #ccc;
    cursor: not-allowed;
    box-shadow: none;
}
```

### 7.2.3 Progress and Results Styling

```css
.progress-section {
    margin: 30px 0;
    text-align: center;
}

.progress-bar {
    width: 100%;
    height: 30px;
    background: rgba(255, 255, 255, 0.2);
    border-radius: 15px;
    overflow: hidden;
    margin-bottom: 15px;
}

.progress-fill {
    height: 100%;
    background: var(--accent-color);
    width: 0%;
    transition: width 0.3s ease;
    box-shadow: 0 0 10px var(--accent-color);
}

.progress-text {
    color: white;
    font-size: 1.1em;
}

.results-section {
    margin-top: 40px;
}

.image-comparison {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 30px;
    margin-bottom: 30px;
}

.image-container {
    background: var(--glass-bg);
    border-radius: 15px;
    padding: 20px;
    text-align: center;
}

.image-container h3 {
    color: white;
    margin-top: 0;
    font-size: 1.3em;
}

.image-container img {
    width: 100%;
    height: auto;
    border-radius: 10px;
    margin-bottom: 15px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
}

.download-btn, .new-image-btn {
    padding: 12px 30px;
    background: var(--primary-color);
    color: white;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    font-size: 1em;
    transition: all 0.3s ease;
}

.download-btn:hover, .new-image-btn:hover {
    background: var(--secondary-color);
    transform: translateY(-2px);
}

.new-image-btn {
    width: 100%;
    padding: 15px;
    font-size: 1.1em;
}

@media (max-width: 768px) {
    .image-comparison {
        grid-template-columns: 1fr;
    }
    
    .upload-label {
        padding: 30px 50px;
    }
}
```

## 7.3 JavaScript Functionality

### 7.3.1 Event Handlers

```javascript
// app.js

const imageInput = document.getElementById('imageInput');
const processBtn = document.getElementById('processBtn');
const progressSection = document.getElementById('progressSection');
const resultsSection = document.getElementById('resultsSection');
const originalImage = document.getElementById('originalImage');
const enhancedImage = document.getElementById('enhancedImage');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const newImageBtn = document.getElementById('newImageBtn');

let selectedFile = null;
let originalPath = null;
let enhancedPath = null;

// File selection handler
imageInput.addEventListener('change', (e) => {
    selectedFile = e.target.files[0];
    if (selectedFile) {
        processBtn.disabled = false;
        updateFileInfo(selectedFile.name);
    }
});

// Drag and drop support
const uploadLabel = document.querySelector('.upload-label');

uploadLabel.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadLabel.style.borderColor = 'var(--accent-color)';
});

uploadLabel.addEventListener('dragleave', () => {
    uploadLabel.style.borderColor = 'var(--glass-border)';
});

uploadLabel.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadLabel.style.borderColor = 'var(--glass-border)';
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        selectedFile = files[0];
        imageInput.files = files;
        processBtn.disabled = false;
        updateFileInfo(selectedFile.name);
    }
});

function updateFileInfo(filename) {
    const fileInfo = document.querySelector('.file-info');
    fileInfo.textContent = `Selected: ${filename}`;
}

// Process button handler
processBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    
    // Hide results, show progress
    resultsSection.style.display = 'none';
    progressSection.style.display = 'block';
    processBtn.disabled = true;
    
    // Simulate progress
    simulateProgress();
    
    try {
        // Create form data
        const formData = new FormData();
        formData.append('file', selectedFile);
        
        // Send request
        const response = await fetch('http://localhost:8000/process', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Processing failed');
        }
        
        const result = await response.json();
        
        // Update progress to 100%
        progressFill.style.width = '100%';
        progressText.textContent = 'Complete!';
        
        // Show results after brief delay
        setTimeout(() => {
            displayResults(result);
        }, 500);
        
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to process image. Please try again.');
        resetUI();
    }
});

function simulateProgress() {
    let progress = 0;
    const interval = setInterval(() => {
        progress += Math.random() * 15;
        if (progress > 90) {
            progress = 90;
            clearInterval(interval);
        }
        progressFill.style.width = `${progress}%`;
        progressText.textContent = `Processing... ${Math.round(progress)}%`;
    }, 500);
}

function displayResults(result) {
    // Hide progress
    progressSection.style.display = 'none';
    
    // Set image sources
    originalPath = `http://localhost:8000${result.original}`;
    enhancedPath = `http://localhost:8000${result.enhanced}`;
    
    originalImage.src = originalPath;
    enhancedImage.src = enhancedPath;
    
    // Show results
    resultsSection.style.display = 'block';
}

function downloadImage(type) {
    const url = type === 'original' ? originalPath : enhancedPath;
    const filename = type === 'original' ? 'original.jpg' : 'enhanced.jpg';
    
    fetch(url)
        .then(response => response.blob())
        .then(blob => {
            const link = document.createElement('a');
            link.href = URL.createObjectURL(blob);
            link.download = filename;
            link.click();
        });
}

newImageBtn.addEventListener('click', () => {
    resetUI();
});

function resetUI() {
    selectedFile = null;
    originalPath = null;
    enhancedPath = null;
    imageInput.value = '';
    processBtn.disabled = true;
    resultsSection.style.display = 'none';
    progressSection.style.display = 'none';
    progressFill.style.width = '0%';
    document.querySelector('.file-info').textContent = 'or drag and drop';
}
```

---

# 8. API Documentation

## 8.1 Endpoint Reference

### 8.1.1 Process Image

**POST** `/process`

Upload and process an underwater image.

**Request:**
- Content-Type: `multipart/form-data`
- Body: Form with `file` field containing image

**Response:**
```json
{
    "success": true,
    "original": "/uploads/image.jpg",
    "enhanced": "/outputs/enhanced_image.jpg",
    "filename": "enhanced_image.jpg"
}
```

**Status Codes:**
- `200`: Success
- `400`: Invalid file type or size
- `500`: Processing error

**Example (curl):**
```bash
curl -X POST http://localhost:8000/process \
  -F "file=@underwater.jpg"
```

**Example (Python):**
```python
import requests

files = {'file': open('underwater.jpg', 'rb')}
response = requests.post('http://localhost:8000/process', files=files)
result = response.json()
print(result)
```

### 8.1.2 Get Original Image

**GET** `/uploads/{filename}`

Retrieve uploaded original image.

**Parameters:**
- `filename`: Name of the uploaded file

**Response:**
- Content-Type: `image/jpeg` (or appropriate image type)
- Body: Image file

**Example:**
```
GET http://localhost:8000/uploads/image.jpg
```

### 8.1.3 Get Enhanced Image

**GET** `/outputs/{filename}`

Retrieve processed enhanced image.

**Parameters:**
- `filename`: Name of the enhanced file

**Response:**
- Content-Type: `image/jpeg`
- Body: Enhanced image file

**Example:**
```
GET http://localhost:8000/outputs/enhanced_image.jpg
```

### 8.1.4 Health Check

**GET** `/health`

Check API health status.

**Response:**
```json
{
    "status": "healthy",
    "version": "1.0.0"
}
```

### 8.1.5 API Documentation

**GET** `/docs`

Interactive Swagger UI documentation.

**GET** `/redoc`

Alternative ReDoc documentation.

## 8.2 Error Handling

### 8.2.1 Error Response Format

```json
{
    "detail": "Error message description"
}
```

### 8.2.2 Common Errors

**400 Bad Request:**
```json
{
    "detail": "Invalid file type. Allowed: {'.jpg', '.jpeg', '.png', '.bmp'}"
}
```

**413 Payload Too Large:**
```json
{
    "detail": "File too large. Maximum size: 50MB"
}
```

**500 Internal Server Error:**
```json
{
    "detail": "Processing failed: [error details]"
}
```

## 8.3 Rate Limiting

Currently no rate limiting implemented. For production deployment, consider:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/process")
@limiter.limit("10/minute")
async def process_image(...):
    ...
```

---

*This is page 1-40 of the comprehensive documentation. Additional sections will continue...*
