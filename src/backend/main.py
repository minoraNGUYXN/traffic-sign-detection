from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from src.app.box_detector import Detector
import numpy as np
import cv2
import base64

app = FastAPI()
detector = Detector()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# API endpoint for processing frames
@app.post("/process_frame")
async def process_frame(file: UploadFile = File(...)):
    try:
        # Read image file content
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return {"error": "Cannot read image", "success": False}

        # Process the frame
        processed_frame, detections_info = detector.process_frame(frame)

        # Encode processed image to base64
        _, buffer = cv2.imencode('.jpg', processed_frame)
        processed_image_b64 = base64.b64encode(buffer).decode('utf-8')

        # Return results
        return {
            "traffic_signs": len(detections_info),
            "detections": [
                {
                    "class_name": detection["class_name"],
                    "confidence": float(detection["confidence"]),
                    "coords": [int(x) for x in detection["bbox"]]
                }
                for detection in detections_info
            ],
            "processed_image": f"data:image/jpeg;base64,{processed_image_b64}",
            "success": True
        }

    except Exception as e:
        return {
            "error": str(e),
            "success": False,
            "traffic_signs": 0,
            "detections": []
        }


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}


# Statistics endpoint
@app.get("/stats")
async def get_stats():
    try:
        stats = detector.get_statistics()
        return {"status": "ok", "statistics": stats}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# Mount static files
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


# Serve index.html at root
@app.get("/")
async def read_index():
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Traffic Sign Detection API", "status": "running"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)