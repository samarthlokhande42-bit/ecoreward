import torch
from ultralytics import YOLO

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA (GPU) Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
else:
    print("Running in CPU mode.")

# Downloads lightweight YOLOv8 nano model checkpoint to verify inference
model = YOLO("yolov8n.pt")
print("YOLOv8 initialized successfully!")