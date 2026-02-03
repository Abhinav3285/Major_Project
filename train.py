from ultralytics import YOLO
import os

def train_model():
    # 1. Load a pre-trained base model (yolov8n is fast, yolov8s is more accurate)
    model = YOLO('yolov8n.pt') 

    # 2. Path to your data configuration
    # Since train.py is in the root, and data.yaml is in the data folder:
    data_path = os.path.join(os.getcwd(), "data", "data.yaml")

    # 3. Start training
    results = model.train(
        data=data_path,
        epochs=10,         # Adjust epochs based on your needs (50-100 is usually good)
        imgsz=640,         # Standard YOLO image size
        batch=8,          # Adjust based on your computer's memory (RAM/GPU)
        name='brain_tumor_run', # This creates the folder results/brain_tumor_run
        project='results',      # Parent folder for results
        exist_ok=True      # Overwrites the folder if it already exists
    )

if __name__ == "__main__":
    train_model()