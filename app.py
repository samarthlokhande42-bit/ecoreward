import os
# Force PyTorch and OpenMP to single thread to minimize RAM allocation
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from fastapi import FastAPI, File, UploadFile, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import torch
import io
import base64
import sqlite3
import gc
from datetime import datetime
import traceback

torch.set_num_threads(1)

app = FastAPI(title="EcoReward Detection API")

@app.middleware("http")
async def force_cors_headers(request: Request, call_next):
    if request.method == "OPTIONS":
        response = Response()
    else:
        response = await call_next(request)
    
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_FILE = "ecoreward.db"

def init_db():
    try:
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
    except Exception as e:
        print(f"DB Init error: {e}")

init_db()

def log_scan(filename: str, count: int, classes_list: list, points: int):
    try:
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
    except Exception as e:
        print(f"Database logging warning: {e}")

# Load model in evaluation mode
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

@app.get("/")
def root():
    return {
        "message": "EcoReward ML Inference API is running.",
        "supported_classes": list(model.names.values())
    }

@app.post("/predict")
async def predict_waste(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Downscale to 480 to keep peak RAM under 300MB
        image.thumbnail((480, 480))

        with torch.no_grad():
            results = model.predict(source=image, conf=0.35, imgsz=480, verbose=False)

        detections = []
        detected_class_names = []
        total_points = 0

        res = results[0]
        if res.boxes is not None:
            for box in res.boxes:
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

        # Generate plot image
        annotated_arr = res.plot()
        annotated_img = Image.fromarray(annotated_arr[..., ::-1])
        
        buf = io.BytesIO()
        annotated_img.save(buf, format="JPEG", quality=75)
        base64_image = base64.b64encode(buf.getvalue()).decode("utf-8")
        annotated_data_url = f"data:image/jpeg;base64,{base64_image}"

        # Clean memory immediately
        del results, annotated_arr, annotated_img, image, image_bytes
        gc.collect()

        log_scan(
            filename=file.filename or "scan.jpg",
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
    except Exception as e:
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/history")
def get_scan_history(limit: int = 10):
    try:
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
    except Exception as e:
        return {
            "status": "success",
            "history_count": 0,
            "history": []
        }