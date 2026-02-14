from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import cv2
import shutil
import uuid
import time
from pathlib import Path
from depth_estimation import DepthEstimator
from sea_thru import sea_thru_pipeline
import config

app = FastAPI(title="Sea-Thru Underwater Image Recovery API")

# Add CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
os.makedirs(config.UPLOAD_DIR, exist_ok=True)
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

# Get the project root directory
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

# Initialize depth estimator (heavy model, load once)
print("Initializing depth estimator...")
depth_estimator = DepthEstimator(model_type=config.MIDAS_MODEL_TYPE)
print("Depth estimator ready!")

# Serve frontend static files
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/")
async def read_index():
    """Serve the main HTML page"""
    return FileResponse(str(FRONTEND_DIR / "index.html"))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": config.MIDAS_MODEL_TYPE}

def validate_file(file: UploadFile) -> tuple[bool, str]:
    """Validate uploaded file"""
    # Check file extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        return False, f"Invalid file type. Allowed: {', '.join(config.ALLOWED_EXTENSIONS)}"
    
    return True, "OK"

@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    """Process uploaded underwater image with Sea-Thru algorithm"""
    start_time = time.time()
    
    # Validate file
    is_valid, message = validate_file(file)
    if not is_valid:
        raise HTTPException(status_code=400, detail=message)
    
    file_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1].lower()
    input_path = os.path.join(config.UPLOAD_DIR, f"{file_id}{ext}")
    output_path = os.path.join(config.OUTPUT_DIR, f"enhanced_{file_id}{ext}")
    depth_path = os.path.join(config.OUTPUT_DIR, f"depth_{file_id}.png")

    try:
        # Save uploaded file
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Check file size
        file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
        if file_size_mb > config.MAX_FILE_SIZE_MB:
            os.remove(input_path)
            raise HTTPException(
                status_code=400, 
                detail=f"File too large ({file_size_mb:.1f}MB). Max size: {config.MAX_FILE_SIZE_MB}MB"
            )
        
        # 1. Estimate depth
        print(f"Estimating depth for {file.filename}...")
        depth_map = depth_estimator.estimate(input_path)
        
        # Save depth map for visualization
        depth_vis = (depth_map * 255).astype('uint8')
        cv2.imwrite(depth_path, depth_vis)
        
        # 2. Apply Sea-Thru
        print("Applying Sea-Thru algorithm...")
        img = cv2.imread(input_path)
        if img is None:
            raise ValueError("Failed to read image. File may be corrupted.")
        
        enhanced_img = sea_thru_pipeline(img, depth_map)
        
        # 3. Save result
        cv2.imwrite(output_path, enhanced_img)
        
        processing_time = time.time() - start_time
        print(f"Processing completed in {processing_time:.2f}s")
        
        return {
            "success": True,
            "original": f"/uploads/{os.path.basename(input_path)}",
            "enhanced": f"/outputs/{os.path.basename(output_path)}",
            "depth": f"/outputs/{os.path.basename(depth_path)}",
            "processing_time": round(processing_time, 2)
        }
    except Exception as e:
        # Clean up files on error
        for path in [input_path, output_path, depth_path]:
            if os.path.exists(path):
                os.remove(path)
        
        error_msg = f"Processing failed: {str(e)}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/uploads/{filename}")
async def get_upload(filename: str):
    """Retrieve uploaded image"""
    file_path = os.path.join(config.UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

@app.get("/outputs/{filename}")
async def get_output(filename: str):
    """Retrieve processed output image"""
    file_path = os.path.join(config.OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

if __name__ == "__main__":
    import uvicorn
    print(f"Starting server on {config.HOST}:{config.PORT}")
    uvicorn.run(
        "main:app",  # Use import string instead of app object for reload
        host=config.HOST, 
        port=config.PORT, 
        reload=config.RELOAD
    )

