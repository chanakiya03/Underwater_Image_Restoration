# Quick Start Guide - Sea-Thru Underwater Image Recovery

## 🚀 30-Second Setup

```bash
# 1. Navigate to project
cd d:\projects\robust_underwater_app

# 2. Install dependencies (one-time)
pip install -r requirements.txt

# 3. Start server
cd backend
python main.py

# 4. Open browser
# Go to: http://localhost:8000
```

## 📸 Quick Test

1. **Upload** any underwater image (JPG, PNG, BMP, TIFF)
2. **Process** - wait 10-30 seconds
3. **Download** the enhanced result

## ⚙️ Configuration

Edit `backend/config.py` to adjust:
- `BETA_COEFFICIENTS` - water attenuation (try different values for different water types)
- `SATURATION_BOOST` - color intensity (default: 1.2)
- `CONTRAST_ALPHA` - contrast enhancement (default: 1.1)

## 🎯 Key Features

✅ Automatic depth estimation  
✅ Physics-based color recovery  
✅ Smart backscatter removal  
✅ Modern web interface  
✅ GPU acceleration (if available)

## 📊 Expected Results

- **Before**: Blue/green tinted, low visibility
- **After**: Natural colors, enhanced clarity, reduced water effects

## 🐛 Troubleshooting

**Server won't start?**
```bash
pip install --upgrade fastapi uvicorn torch opencv-python
```

**Processing too slow?**
- Use smaller images (resize to 1080p)
- Switch to `MiDaS_small` in config.py

**Colors look off?**
- Adjust `BETA_COEFFICIENTS` in config.py
- Try increasing `SATURATION_BOOST`

## 📚 Full Documentation

See [README.md](file:///d:/projects/robust_underwater_app/README.md) for complete documentation.
