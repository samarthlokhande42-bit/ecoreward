from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import torch
import io
import base64
import sqlite3
from datetime import datetime

app = FastAPI(title="EcoReward Detection API")

# Enable CORS for local testing and Netlify deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ecoreward-system.netlify.app",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:3000",
        "http://127.0.0.1:8000",
        "*"
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- DATABASE SETUP (SQLite) -----------------
DB_FILE = "ecoreward.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scan_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            timestamp TEXT,
            items_count INTEGER,
            detected_classes TEXT,
            total_points INTEGER
        )
    """)
    conn.commit()
    conn.close()

init_db()

def log_scan(filename: str, count: int, classes_list: list, points: int):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO scan_history (filename, timestamp, items_count, detected_classes, total_points)
        VALUES (?, ?, ?, ?, ?)
    """, (
        filename,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        count,
        ", ".join(classes_list) if classes_list else "None",
        points
    ))
    conn.commit()
    conn.close()

# ----------------- MODEL & REWARD CONFIG -----------------
model = YOLO("best.pt")

REWARD_MAP = {
    "pet bottle": 10,
    "hdpe plastic": 15,
    "single-layer plastic": 5,
    "multi-layer plastic": 5,
    "squeeze-tube": 5,
    "uht-box": 8,
    "single-use-plastic": 2
}

# ----------------- ENDPOINTS -----------------

@app.get("/")
def root():
    return {
        "message": "EcoReward ML Inference API is running.",
        "supported_classes": list(model.names.values())
    }

@app.post("/predict")
async def predict_waste(file: UploadFile = File(...)):
    # 1. Read uploaded image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Resize high-res photos to prevent Render 512MB RAM OOM crash
    image.thumbnail((640, 640))

    # 2. Run YOLO inference inside torch.no_grad()
    with torch.no_grad():
        results = model(image, conf=0.4, imgsz=640)

    detections = []
    detected_class_names = []
    total_points = 0

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        confidence = float(box.conf[0])

        points = REWARD_MAP.get(class_name.lower().strip(), 0)
        total_points += points

        detections.append({
            "class": class_name,
            "confidence": round(confidence, 3),
            "points_earned": points,
            "box": [round(coord, 1) for coord in box.xyxy[0].tolist()]
        })
        detected_class_names.append(class_name)

    # 3. Generate Annotated Image (Base64)
    annotated_arr = results[0].plot()
    annotated_img = Image.fromarray(annotated_arr[..., ::-1])  # BGR to RGB
    
    buf = io.BytesIO()
    annotated_img.save(buf, format="JPEG", quality=80)
    base64_image = base64.b64encode(buf.getvalue()).decode("utf-8")
    annotated_data_url = f"data:image/jpeg;base64,{base64_image}"

    # 4. Save record to SQLite database
    log_scan(
        filename=file.filename,
        count=len(detections),
        classes_list=detected_class_names,
        points=total_points
    )

    return {
        "status": "success",
        "detections_count": len(detections),
        "detections": detections,
        "total_reward_points": total_points,
        "annotated_image": annotated_data_url
    }

@app.get("/history")
def get_scan_history(limit: int = 10):
    """Retrieve the most recent user scans and reward records."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM scan_history 
        ORDER BY id DESC 
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()

    history = [dict(row) for row in rows]
    return {
        "status": "success",
        "history_count": len(history),
        "history": history
    }