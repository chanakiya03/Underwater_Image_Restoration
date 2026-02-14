# Configuration, Testing, Deployment & User Guide

# 9. Configuration Guide

## 9.1 Configuration File Overview

The `config.py` file centralizes all system settings:

```python
# config.py

import os

# ---------- File Upload Settings ----------
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

# ---------- Directory Settings ----------
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
STATIC_DIR = "frontend"

# Create directories if they don't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- MiDaS Depth Estimation Settings ----------
MIDAS_MODEL = "MiDaS_small"  # Options: "MiDaS_small", "DPT_Hybrid", "DPT_Large"

# ---------- Sea-Thru Algorithm Parameters ----------
BETA_COEFFICIENTS = {
    'blue': 0.3,    # Blue channel attenuation
    'green': 0.15,  # Green channel attenuation  
    'red': 0.08     # Red channel attenuation
}

BACKSCATTER_ESTIMATION_METHOD = "dark_pixels"
DARK_PIXEL_FRACTION = 0.001  # 0.1% darkest pixels

# ---------- Image Processing Parameters ----------
ENHANCE_CONTRAST = True
ENHANCE_SATURATION = True
SATURATION_BOOST = 1.25  # Multiplier for saturation (1.0 = no change)
CONTRAST_ALPHA = 1.1     # Contrast enhancement factor

# ---------- HD Quality Enhancement ----------
ADVANCED_DENOISING = True      # Use Non-Local Means denoising
DETAIL_ENHANCEMENT = True       # High-pass filter for detail
EDGE_SHARPENING = True         # Unsharp mask sharpening
ENABLE_SUPER_RESOLUTION = False  # 2x upscaling (slow!)

# ---------- Server Settings ----------
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
RELOAD = True  # Auto-reload on code changes (development only)
```

## 9.2 Parameter Tuning Guide

### 9.2.1 Saturation Boost

Controls color vibrancy:

```python
SATURATION_BOOST = 1.0   # No boost (natural)
SATURATION_BOOST = 1.25  # Balanced (recommended)
SATURATION_BOOST = 1.5   # Vibrant
SATURATION_BOOST = 2.0   # Very vibrant (may look artificial)
```

**When to adjust:**
- Images too dull → Increase to 1.3-1.4
- Images oversaturated → Decrease to 1.1-1.2

### 9.2.2 Beta Coefficients

Controls depth-dependent color restoration:

```python
BETA_COEFFICIENTS = {
    'blue': 0.3,    # ↑ Remove more blue
    'green': 0.15,  # ↑ Remove more green
    'red': 0.08     # ↑ Boost red less (use lower values)
}
```

**Tuning tips:**
- Too blue → Increase blue/green coefficients
- Too warm/red → Decrease red coefficient
- Depth effects too strong → Scale all down proportionally

### 9.2.3 MiDaS Model Selection

Trade-off between speed and accuracy:

| Model | Speed | Accuracy | Memory | Use Case |
|-------|-------|----------|--------|----------|
| MiDaS_small | Fast | Good | Low | Production, real-time |
| DPT_Hybrid | Medium | Better | Medium | Balance |
| DPT_Large | Slow | Best | High | Quality-critical |

```python
# Fast processing
MIDAS_MODEL = "MiDaS_small"

# Best quality
MIDAS_MODEL = "DPT_Large"
```

### 9.2.4 HD Enhancement Options

Enable/disable individual enhancement steps:

```python
# Recommended for production (balanced)
ADVANCED_DENOISING = True      # Removes noise
DETAIL_ENHANCEMENT = True       # Enhances fine details
EDGE_SHARPENING = True         # Sharpens edges
ENABLE_SUPER_RESOLUTION = False  # 2x upscaling (very slow!)

# Speed-optimized (faster but lower quality)
ADVANCED_DENOISING = False     # Use faster bilateral filter
DETAIL_ENHANCEMENT = False
EDGE_SHARPENING = False

# Quality-optimized (slower but best results)
ADVANCED_DENOISING = True
DETAIL_ENHANCEMENT = True
EDGE_SHARPENING = True
ENABLE_SUPER_RESOLUTION = True  # For 4K output
```

## 9.3 Environment-Specific Configuration

### 9.3.1 Development Configuration

```python
# config.py (development)

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
RELOAD = True
DEBUG = True
MAX_FILE_SIZE = 50 * 1024 * 1024

# Faster processing
MIDAS_MODEL = "MiDaS_small"
ENABLE_SUPER_RESOLUTION = False
```

### 9.3.2 Production Configuration

```python
# config.py (production)

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8080
RELOAD = False
DEBUG = False
MAX_FILE_SIZE = 20 * 1024 * 1024  # Stricter limit

# Balanced quality/speed
MIDAS_MODEL = "DPT_Hybrid"
ENABLE_SUPER_RESOLUTION = False

# Enable logging
import logging
logging.basicConfig(level=logging.INFO)
```

---

# 10. Testing and Validation

## 10.1 Unit Testing

### 10.1.1 Test Structure

```python
# tests/test_sea_thru.py

import pytest
import numpy as np
import cv2
from backend.sea_thru import (
    estimate_backscatter_from_dark_pixels,
    recover_colors,
    sea_thru_pipeline
)

class TestBackscatterEstimation:
    def test_dark_pixel_detection(self):
        # Create test image with known dark pixels
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        img[0:10, 0:10] = [10, 20, 30]  # Dark region
        
        backscatter = estimate_backscatter_from_dark_pixels(img, fraction=0.01)
        
        # Backscatter should be close to dark pixel values
        assert backscatter[0] < 50  # Blue
        assert backscatter[1] < 50  # Green
        assert backscatter[2] < 50  # Red
    
    def test_backscatter_range(self):
        # Test with real underwater image
        img = cv2.imread('test_data/underwater1.jpg')
        backscatter = estimate_backscatter_from_dark_pixels(img)
        
        # Backscatter should be in valid range
        assert np.all(backscatter >= 0)
        assert np.all(backscatter <= 255)

class TestColorRecovery:
    def test_color_recovery_basic(self):
        # Create synthetic underwater image
        img = np.ones((100, 100, 3), dtype=np.uint8)
        img[:, :, 0] = 150  # Heavy blue
        img[:, :, 1] = 100  # Medium green
        img[:, :, 2] = 50   # Low red
        
        depth = np.ones((100, 100)) * 0.5
        B_inf = np.array([100, 80, 60])
        illuminant = np.array([255, 255, 255])
        beta = np.array([0.3, 0.15, 0.08])
        
        recovered = recover_colors(img, depth, B_inf, illuminant, beta)
        
        # Red should be boosted
        assert recovered[:, :, 2].mean() > img[:, :, 2].mean()
        
        # Blue should be reduced
        assert recovered[:, :, 0].mean() < img[:, :, 0].mean()
    
    def test_output_range(self):
        img = cv2.imread('test_data/underwater1.jpg')
        depth = np.random.rand(*img.shape[:2])
        B_inf = np.array([100, 80, 60])
        illuminant = np.array([255, 255, 255])
        beta = np.array([0.3, 0.15, 0.08])
        
        recovered = recover_colors(img, depth, B_inf, illuminant, beta)
        
        # Output should be uint8 [0, 255]
        assert recovered.dtype == np.uint8
        assert np.all(recovered >= 0)
        assert np.all(recovered <= 255)

class TestEndToEnd:
    def test_pipeline_execution(self):
        img = cv2.imread('test_data/underwater1.jpg')
        depth = np.random.rand(*img.shape[:2])
        
        result = sea_thru_pipeline(img, depth)
        
        # Result should have same shape
        assert result.shape == img.shape
        assert result.dtype == np.uint8
    
    def test_multiple_images(self):
        test_images = [
            'test_data/underwater1.jpg',
            'test_data/underwater2.jpg',
            'test_data/underwater3.jpg'
        ]
        
        for img_path in test_images:
            img = cv2.imread(img_path)
            depth = np.random.rand(*img.shape[:2])
            result = sea_thru_pipeline(img, depth)
            
            assert result is not None
            assert result.shape == img.shape

# Run tests
if __name__ == "__main__":
    pytest.main([__file__, '-v'])
```

### 10.1.2 Running Tests

```bash
# Install pytest
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=backend --cov-report=html

# Run specific test file
pytest tests/test_sea_thru.py -v

# Run specific test
pytest tests/test_sea_thru.py::TestBackscatterEstimation::test_dark_pixel_detection -v
```

## 10.2 Integration Testing

### 10.2.1 API Testing

```python
# tests/test_api.py

import pytest
from fastapi.testclient import TestClient
from backend.main import app
import os

client = TestClient(app)

class TestAPI:
    def test_health_endpoint(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_process_image_success(self):
        # Prepare test image
        test_image_path = "test_data/underwater1.jpg"
        
        with open(test_image_path, "rb") as f:
            response = client.post(
                "/process",
                files={"file": ("test.jpg", f, "image/jpeg")}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "original" in data
        assert "enhanced" in data
    
    def test_invalid_file_type(self):
        # Try to upload invalid file
        response = client.post(
            "/process",
            files={"file": ("test.txt", b"not an image", "text/plain")}
        )
        
        assert response.status_code == 400
   
    def test_no_file(self):
        response = client.post("/process")
        assert response.status_code == 422  # Unprocessable entity
   
    def test_get_uploaded_image(self):
        # First upload an image
        test_image_path = "test_data/underwater1.jpg"
        
        with open(test_image_path, "rb") as f:
            upload_response = client.post(
                "/process",
                files={"file": ("test.jpg", f, "image/jpeg")}
            )
        
        data = upload_response.json()
        original_path = data["original"]
        
        # Then retrieve it
        response = client.get(original_path)
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("image/")
```

## 10.3 Performance Testing

### 10.3.1 Benchmark Script

```python
# tests/benchmark.py

import time
import cv2
import numpy as np
from backend.depth_estimation import DepthEstimator
from backend.sea_thru import sea_thru_pipeline

def benchmark_depth_estimation():
    estimator = DepthEstimator()
    test_images = [
        "test_data/image_small.jpg",   # 640x480
        "test_data/image_medium.jpg",  # 1920x1080
        "test_data/image_large.jpg"    # 3840x2160
    ]
    
    results = []
    for img_path in test_images:
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        
        start = time.time()
        depth = estimator.estimate(img_path)
        elapsed = time.time() - start
        
        results.append({
            'resolution': f"{w}x{h}",
            'time': elapsed,
            'fps': 1/elapsed
        })
    
    print("\nDepth Estimation Benchmark:")
    print("-" * 50)
    for r in results:
        print(f"{r['resolution']:15} | {r['time']:.2f}s | {r['fps']:.2f} FPS")
    
    return results

def benchmark_sea_thru():
    test_images = [
        "test_data/image_small.jpg",
        "test_data/image_medium.jpg",
        "test_data/image_large.jpg"
    ]
    
    results = []
    for img_path in test_images:
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        depth = np.random.rand(h, w)
        
        start = time.time()
        enhanced = sea_thru_pipeline(img, depth)
        elapsed = time.time() - start
        
        results.append({
            'resolution': f"{w}x{h}",
            'time': elapsed
        })
    
    print("\nSea-Thru Processing Benchmark:")
    print("-" * 50)
    for r in results:
        print(f"{r['resolution']:15} | {r['time']:.2f}s")
    
    return results

if __name__ == "__main__":
    print("Running Performance Benchmarks...")
    benchmark_depth_estimation()
    benchmark_sea_thru()
```

## 10.4 Quality Validation

### 10.4.1 Metrics

Common image quality metrics:

```python
# tests/quality_metrics.py

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def calculate_psnr(original, enhanced):
    """Peak Signal-to-Noise Ratio"""
    return peak_signal_noise_ratio(original, enhanced)

def calculate_ssim(original, enhanced):
    """Structural Similarity Index"""
    return structural_similarity(
        original, enhanced,
        multichannel=True,
        channel_axis=2
    )

def calculate_color_balance(img):
    """Measure color balance (mean of each channel)"""
    return {
        'blue': np.mean(img[:,:,0]),
        'green': np.mean(img[:,:,1]),
        'red': np.mean(img[:,:,2])
    }

def calculate_contrast(img):
    """Standard deviation as contrast measure"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.std(gray)

def evaluate_enhancement(original_path, enhanced_path):
    """Comprehensive quality evaluation"""
    original = cv2.imread(original_path)
    enhanced = cv2.imread(enhanced_path)
    
    metrics = {
        'psnr': calculate_psnr(original, enhanced),
        'ssim': calculate_ssim(original, enhanced),
        'original_balance': calculate_color_balance(original),
        'enhanced_balance': calculate_color_balance(enhanced),
        'original_contrast': calculate_contrast(original),
        'enhanced_contrast': calculate_contrast(enhanced)
    }
    
    return metrics
```

---

# 11. Deployment Guide

## 11.1 Local Deployment

### 11.1.1 Quick Start

```bash
# Clone repository
git clone <repository-url>
cd robust_underwater_app

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run server
python main.py

# Access application
# Open browser: http://localhost:8000
```

### 11.1.2 Development Mode

```bash
# Run with auto-reload
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000

# With custom config
export MIDAS_MODEL=DPT_Hybrid
python main.py
```

## 11.2 Docker Deployment

### 11.2.1 Dockerfile

```dockerfile
# Dockerfile

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create necessary directories
RUN mkdir -p uploads outputs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run application
CMD ["python", "main.py"]
```

### 11.2.2 Docker Compose

```yaml
# docker-compose.yml

version: '3.8'

services:
  underwater-enhancement:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./uploads:/app/uploads
      - ./outputs:/app/outputs
    environment:
      - MIDAS_MODEL=MiDaS_small
      - MAX_FILE_SIZE=52428800
    restart: unless-stopped
    mem_limit: 4g
    cpus: 2.0
```

### 11.2.3 Build and Run

```bash
# Build image
docker build -t underwater-enhancement .

# Run container
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/outputs:/app/outputs \
  --name underwater-app \
  underwater-enhancement

# Using docker-compose
docker-compose up -d

# View logs
docker logs -f underwater-app

# Stop
docker stop underwater-app
docker-compose down
```

## 11.3 Cloud Deployment

### 11.3.1 AWS EC2

**1. Launch EC2 Instance:**
- AMI: Ubuntu 22.04 LTS
- Instance type: t3.medium (minimum)
- Security group: Allow ports 22 (SSH), 8000 (HTTP)

**2. Connect and Setup:**

```bash
# SSH into instance
ssh -i key.pem ubuntu@<ec2-public-ip>

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install -y python3-pip python3-venv git

# Clone repository
git clone <repository-url>
cd robust_underwater_app

# Setup application
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run with systemd (production)
sudo nano /etc/systemd/system/underwater-app.service
```

**3. Systemd Service File:**

```ini
[Unit]
Description=Underwater Image Enhancement API
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/robust_underwater_app
Environment="PATH=/home/ubuntu/robust_underwater_app/venv/bin"
ExecStart=/home/ubuntu/robust_underwater_app/venv/bin/python main.py
Restart=always

[Install]
WantedBy=multi-user.target
```

**4. Start Service:**

```bash
sudo systemctl daemon-reload
sudo systemctl enable underwater-app
sudo systemctl start underwater-app
sudo systemctl status underwater-app
```

### 11.3.2 Google Cloud Platform (GCP)

**Using Cloud Run:**

```bash
# Install gcloud CLI
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login

# Set project
gcloud config set project PROJECT_ID

# Build and push container
gcloud builds submit --tag gcr.io/PROJECT_ID/underwater-enhancement

# Deploy to Cloud Run
gcloud run deploy underwater-enhancement \
  --image gcr.io/PROJECT_ID/underwater-enhancement \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300 \
  --allow-unauthenticated
```

### 11.3.3 Azure App Service

```bash
# Install Azure CLI
# https://docs.microsoft.com/cli/azure/install-azure-cli

# Login
az login

# Create resource group
az group create --name underwater-rg --location eastus

# Create App Service plan
az appservice plan create \
  --name underwater-plan \
  --resource-group underwater-rg \
  --sku B2 \
  --is-linux

# Create web app
az webapp create \
  --resource-group underwater-rg \
  --plan underwater-plan \
  --name underwater-enhancement \
  --deployment-container-image-name underwater-enhancement:latest

# Configure app
az webapp config appsettings set \
  --resource-group underwater-rg \
  --name underwater-enhancement \
  --settings WEBSITES_PORT=8000
```

## 11.4 Production Considerations

### 11.4.1 Nginx Reverse Proxy

```nginx
# /etc/nginx/sites-available/underwater-app

server {
    listen 80;
    server_name your-domain.com;

    client_max_body_size 50M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeout settings
        proxy_connect_timeout 300;
        proxy_send_timeout 300;
        proxy_read_timeout 300;
    }
}
```

Enable site:

```bash
sudo ln -s /etc/nginx/sites-available/underwater-app /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### 11.4.2 SSL/HTTPS with Let's Encrypt

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal is configured automatically
sudo certbot renew --dry-run
```

### 11.4.3 Monitoring and Logging

```python
# Add to main.py

import logging
from logging.handlers import RotatingFileHandler

# Configure logging
handler = RotatingFileHandler(
    'app.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# Use in endpoints
@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    logger.info(f"Processing image: {file.filename}")
    try:
        # ... processing ...
        logger.info(f"Successfully processed: {file.filename}")
    except Exception as e:
        logger.error(f"Error processing {file.filename}: {str(e)}")
```

---

*Documentation continues with User Guide, Developer Guide, Troubleshooting, References, and Appendices...*
