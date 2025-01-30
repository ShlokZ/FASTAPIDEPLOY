from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image, ImageDraw
import io
import os
import uuid
from pathlib import Path

app = FastAPI()

# Define paths
STATIC_DIR = "app/static"
OUTPUT_DIR = "app/static/output"
MODEL_PATH = "app/model/best.pt"

# Ensure necessary directories exist
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Verify that the model file exists before loading
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

# Load YOLO model
model = YOLO(MODEL_PATH)

# Mount static directory for serving images
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.post("/detect/")
async def detect(file: UploadFile = File()):
    try:
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
        unique_filename = f"{uuid.uuid4().hex}.jpg"
        output_image_path = Path(OUTPUT_DIR) / unique_filename
        image.save(output_image_path)

        # Return JSON response with detections and image URL
        return {
            "detections": output,
            "image_url": f"/static/output/{unique_filename}"
        }
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/")
async def root():
    return {"message": "YOLO Object Detection API is Running"}
