# Installation Fix for Windows (No Compiler Required)

## The Problem
NumPy is trying to build from source but can't find a C compiler (Visual Studio, gcc, etc.)

## ✅ Quick Fix - Install Step by Step

Run these commands **one at a time** in your activated virtual environment:

```bash
# 1. Core web framework (fast)
pip install fastapi uvicorn[standard] python-multipart

# 2. NumPy - use precompiled wheel (IMPORTANT)
pip install numpy --only-binary :all:

# 3. OpenCV
pip install opencv-python --only-binary :all:

# 4. Pillow
pip install Pillow --only-binary :all:

# 5. PyTorch (this is BIG - ~2GB download, be patient)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 6. PyTorch Image Models (required by MiDaS)
pip install timm
```

## Why Step 5 Uses CPU Version?

The CPU version of PyTorch is smaller (~2GB vs ~5GB) and works fine for this app. If you have an NVIDIA GPU and want faster processing, use:

```bash
# GPU version (CUDA 11.8) - only if you have NVIDIA GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## ✅ Verify Installation

After all packages are installed, test:

```bash
python -c "import torch; import cv2; import fastapi; import numpy; print('✅ Success!')"
```

## 🚀 Then Start the Server

```bash
cd backend
python main.py
```

---

## Alternative: Use Conda (If Available)

If you have Anaconda/Miniconda:

```bash
conda create -n underwater python=3.11
conda activate underwater
conda install numpy opencv pillow pytorch torchvision cpuonly -c pytorch
pip install fastapi uvicorn[standard] python-multipart
```

---

**Note**: I've already started installing numpy/opencv/pillow with the `--only-binary` flag. Check if that command completes successfully before trying the step-by-step approach above.
