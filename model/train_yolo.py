from ultralytics import YOLO
import os

def train_brain_tumor_model():
    # 1. Load a pretrained YOLOv8 nano model (good for speed and efficiency)
    model = YOLO('yolov8n.pt') 

    # 2. Define the path to your data.yaml
    yaml_path = os.path.join('../data/data.yaml')

    # 3. Train the model
    # epochs: Number of passes over the data
    # imgsz: Image size (MRI standard is often 640)
    results = model.train(
        data=yaml_path,
        epochs=50,
        imgsz=640,
        batch=16,
        project='../results',
        name='brain_tumor_run'
    )
    
    print("Training complete. Model saved in ../results/brain_tumor_run/weights/best.pt")

if __name__ == "__main__":
    train_brain_tumor_model()