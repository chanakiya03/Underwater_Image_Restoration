# 🌊 Sea-Thru: Underwater Image Color Recovery

A web application that removes water effects from underwater images using the **Sea-Thru algorithm** from the CVPR 2019 paper: *"Sea-Thru: A Method for Removing Water from Underwater Images"*.

![Sea-Thru Demo](https://via.placeholder.com/800x400.png?text=Before+%7C+After+Comparison)

## ✨ Features

- **Physics-based Color Recovery**: Uses the Sea-Thru physical model for accurate underwater image restoration
- **Automatic Depth Estimation**: Leverages MiDaS deep learning model for depth map generation
- **Smart Backscatter Estimation**: Analyzes darkest pixels in farthest regions
- **Per-Channel Attenuation**: Calculates wavelength-dependent attenuation coefficients
- **Modern UI**: Glassmorphic design with side-by-side comparison
- **Real-time Processing**: Fast GPU-accelerated processing (when available)
- **Download Results**: Save enhanced images directly

## 🧠 How It Works

The Sea-Thru algorithm models underwater image formation:

```
I(x) = J(x) × e^(-βD×d) + B∞ × (1 - e^(-βB×d))
```

Where:
- **I(x)**: Observed underwater image
- **J(x)**: True scene radiance (recovered)
- **βD**: Direct signal attenuation coefficient
- **βB**: Backscatter attenuation coefficient
- **d**: Distance to object (from depth map)
- **B∞**: Veiling light (backscatter at infinity)

### Processing Steps

1. **Depth Estimation**: MiDaS generates a relative depth map
2. **Backscatter Estimation**: Analyzes darkest pixels in distant regions
3. **Illuminant Calculation**: Depth-weighted gray-world assumption
4. **Attenuation Coefficients**: Estimates per-channel β values
5. **Color Recovery**: Inverts the physical model to recover true colors
6. **Enhancement**: Optional contrast and saturation boost

## 📋 Requirements

- Python 3.8+
- CUDA-capable GPU (optional, CPU works but slower)
- Modern web browser

## 🚀 Installation

### 1. Clone or Download

```bash
cd d:\projects\robust_underwater_app
```

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: First run will download the MiDaS model (~100MB) which is cached locally.

## 🎮 Usage

### Start the Server

```bash
cd backend
python main.py
```

The server will start on `http://localhost:8000`

### Use the Web Interface

1. Open your browser to `http://localhost:8000`
2. Click "Choose an underwater image" and select your image
3. Click "Process Image"
4. Wait for processing (10-30 seconds depending on image size)
5. View the enhanced result side-by-side with the original
6. Download the result using the "Download Result" button

### Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)

**Max file size**: 10MB

## 🔧 Configuration

Edit `backend/config.py` to customize:

```python
# Model selection
MIDAS_MODEL_TYPE = "MiDaS_small"  # or "DPT_Large", "DPT_Hybrid"

# Attenuation coefficients
BETA_COEFFICIENTS = {
    'blue': 0.5,    # Adjust for water clarity
    'green': 0.2,
    'red': 0.1
}

# Post-processing
ENHANCE_CONTRAST = True
CONTRAST_ALPHA = 1.1
SATURATION_BOOST = 1.2
```

## 🧪 Testing

### Verify Sea-Thru Algorithm

```bash
python verify_sea_thru.py
```

This will process a sample image and save outputs to `test_outputs/`.

### Health Check

```bash
curl http://localhost:8000/health
```

Expected response: `{"status": "healthy", "model": "MiDaS_small"}`

## 📁 Project Structure

```
robust_underwater_app/
├── backend/
│   ├── main.py              # FastAPI server
│   ├── sea_thru.py          # Sea-Thru algorithm implementation
│   ├── depth_estimation.py  # MiDaS depth estimator
│   └── config.py            # Configuration settings
├── frontend/
│   ├── index.html           # Main UI
│   ├── style.css            # Glassmorphic styling
│   └── app.js               # Upload/processing logic
├── requirements.txt         # Python dependencies
├── .gitignore
└── README.md
```

## 🎨 API Endpoints

### `POST /process`

Upload and process an underwater image.

**Request**:
- Content-Type: `multipart/form-data`
- Body: `file` (image file)

**Response**:
```json
{
  "success": true,
  "original": "/uploads/abc123.jpg",
  "enhanced": "/outputs/enhanced_abc123.jpg",
  "depth": "/outputs/depth_abc123.png",
  "processing_time": 12.34
}
```

### `GET /health`

Check server status.

**Response**:
```json
{
  "status": "healthy",
  "model": "MiDaS_small"
}
```

## 🐛 Troubleshooting

### MiDaS Model Download Fails

**Solution**: Manually download and cache:
```bash
python -c "import torch; torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', trust_repo=True)"
```

### CUDA Out of Memory

**Solution**: Switch to CPU or smaller model in `config.py`:
```python
MIDAS_MODEL_TYPE = "MiDaS_small"  # Smallest, fastest
```

### Poor Quality Results

**Solutions**:
1. Try adjusting `BETA_COEFFICIENTS` in `config.py` for your water type
2. Increase `SATURATION_BOOST` for more vibrant colors
3. Use `DPT_Large` model for better depth estimation (slower)

## 📚 References

- **Sea-Thru Paper**: Akkaynak & Treibitz, "Sea-Thru: A Method for Removing Water from Underwater Images", CVPR 2019
  - [Paper PDF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Akkaynak_Sea-Thru_A_Method_for_Removing_Water_From_Underwater_Images_CVPR_2019_paper.pdf)
  
- **MiDaS Depth Estimation**: Ranftl et al., "Towards Robust Monocular Depth Estimation", 2020
  - [GitHub](https://github.com/isl-org/MiDaS)

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- Batch processing support
- Real-time video processing
- Parameter tuning UI sliders
- More depth estimation models
- Mobile app version

## 📄 License

This project is for educational purposes. Please cite the original Sea-Thru paper if using this in research.

## 🙏 Acknowledgments

- Derya Akkaynak and Tali Treibitz for the Sea-Thru algorithm
- Intel ISL for the MiDaS depth estimation model
- FastAPI and PyTorch communities

---

**Made with 🌊 for underwater photography enthusiasts**
####################################

activate the venv file :

  cmd:  cd robust_underwater_app
  cmd:  venv\Scripts\activate



  cmd: install requirements.txt // (if any error )

           ( or )

  cmd:use install_fix.md follow the instructions


  cmd:cd backend

 run main.py
  cmd: python manage.py runserver 


    open  in browser : http://localhost:8000    copya and past in browser      

   

###
one image processing time: 10-30 seconds
run batch_process.py

  cmd: python batch_process.py

  seletc second options:

  1. that one image path  example : "D:\projects\robust_underwater_app\test\test.jpg" 
      like this in test.jpg image path 

  2:output image path: example : "D:\projects\robust_underwater_app\backend\d_image_enhanced"  like this but you path d_image_enhanced folder path 
