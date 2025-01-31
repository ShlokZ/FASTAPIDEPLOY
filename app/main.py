from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import io
import numpy as np
import uuid
from PIL import Image, ImageDraw


app = FastAPI()

# Load the YOLO model
model = YOLO("app/model/best.pt")

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    # Load the uploaded image
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")

    # Run YOLO inference
    results = model(image)
    detections = results[0].boxes.data.cpu().numpy()

    # Prepare output without drawing on the image
    output = []
    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection
        output.append({
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
            "confidence": float(confidence),
            "class": int(class_id)
        })

    # Return JSON response with detections
    return {"detections": output}
