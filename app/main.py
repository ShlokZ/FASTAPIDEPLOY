from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image, ImageDraw
import io
import numpy as np
from pathlib import Path
import uuid

app = FastAPI()

# Mount static directory for serving images
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Load the YOLO model
model = YOLO("app/model/best.pt")

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    # Load the uploaded image
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")

    # Run YOLO inference
    results = model(image)
    detections = results[0].boxes.data.cpu().numpy()

    # Draw bounding boxes on the image
    draw = ImageDraw.Draw(image)
    output = []
    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection
        # Draw the bounding box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        # Add class label and confidence score
        draw.text((x1, y1 - 10), f"Class {int(class_id)}: {confidence:.2f}", fill="red")
        # Add to output list
        output.append({
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
            "confidence": float(confidence),
            "class": int(class_id)
        })

    # Save the image with a unique filename
    output_dir = Path("app/static/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    unique_filename = f"{uuid.uuid4().hex}.jpg"
    output_image_path = output_dir / unique_filename
    image.save(output_image_path)

    # Return JSON response with detections and image URL
    return {
        "detections": output,
        "image_url": f"/static/output/{unique_filename}"  # Return the unique image path
    }
