import cv2
import numpy as np
from ultralytics import YOLO
import os

# 1. Dynamically find the project root
# This gets the directory where this script is, then goes up one level to 'NeuroScanVision'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'results', 'brain_tumor_seg4', 'weights', 'best.pt')

# Load the model using the absolute path
model = YOLO(MODEL_PATH)

def get_diagnosis(img_path):
    if not os.path.exists(img_path):
        return f"Error: Image not found at {img_path}"

    results = model(img_path)[0]
    
    if not results.masks:
        return {
            "Type": "None",
            "Confidence": "N/A",
            "Area": "0.00 mm²",
            "Clinical Stage": "Healthy"
        }

    # Get details for the first detected tumor
    tumor_type = results.names[int(results.boxes.cls[0])]
    confidence = results.boxes.conf[0]
    
    # Calculate Area using the Mask Polygon
    mask_coords = results.masks.xy[0]
    polygon = np.array(mask_coords, dtype=np.int32)
    pixel_area = cv2.contourArea(polygon)
    
    # Conversion: Assuming 1 pixel = 0.1 mm (Standard MRI resolution often ~0.1-0.2)
    # This means 1 pixel area = 0.01 mm²
    area_mm2 = pixel_area * 0.01 
    
    # Staging Logic
    if area_mm2 < 100:
        stage = "Stage I (Benign/Small)"
    elif area_mm2 < 400:
        stage = "Stage II (Intermediate)"
    else:
        stage = "Stage III (Large/Malignant)"

    return {
        "Type": tumor_type.capitalize(),
        "Confidence": f"{confidence:.2%}",
        "Area": f"{area_mm2:.2f} mm²",
        "Clinical Stage": stage
    }

# --- RUN ON TEST IMAGE ---
# Use an absolute path to be safe
TEST_IMAGE_PATH = os.path.join(BASE_DIR, 'data', 'yolo_segmentation', 'images', 'test', 'Tr-gl_0014.jpg')

report = get_diagnosis(TEST_IMAGE_PATH)

if isinstance(report, dict):
    print("\n" + "="*40)
    print("       🏥 NEUROSCANVISION REPORT       ")
    print("="*40)
    for key, value in report.items():
        print(f"{key:<15}: {value}")
    print("="*40)
else:
    print(report)