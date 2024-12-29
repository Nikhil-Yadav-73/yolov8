from ultralytics import YOLO
import os

data_path = r"C:\Users\Nikhil\Desktop\venv\ultralytics\dataset\data.yaml"

# Load YOLO model and train
model = YOLO('yolov8n.pt')
model.train(data=data_path, epochs=50, imgsz=640)
