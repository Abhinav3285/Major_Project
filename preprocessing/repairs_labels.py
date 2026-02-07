import cv2
import os
import numpy as np
from tqdm import tqdm

def get_project_root():
    current_path = os.path.abspath(__file__)
    while os.path.basename(current_path) != "NeuroScanVision":
        parent = os.path.dirname(current_path)
        if parent == current_path: return os.path.dirname(os.path.abspath(__file__))
        current_path = parent
    return current_path

BASE_DIR = get_project_root()
DATA_ROOT = os.path.join(BASE_DIR, "data", "yolo_dataset")

def get_smart_bbox(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_cnt)
        return (x + w/2)/img.shape[1], (y + h/2)/img.shape[0], w/img.shape[1], h/img.shape[0]
    return 0.5, 0.5, 0.15, 0.15

def get_class_id(filename):
    name = filename.lower()
    if "glioma" in name: return 0
    if "meningioma" in name: return 1
    if "pituitary" in name: return 2
    return 3

def run_fix():
    print(f"🛠️ Fixing Labels in: {DATA_ROOT}")
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(DATA_ROOT, "images", split)
        lbl_dir = os.path.join(DATA_ROOT, "labels", split)
        
        if not os.path.exists(img_dir): continue
        os.makedirs(lbl_dir, exist_ok=True)

        for img_name in tqdm(os.listdir(img_dir), desc=f"Fixing {split}"):
            if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                img = cv2.imread(os.path.join(img_dir, img_name))
                if img is None: continue
                
                cid = get_class_id(img_name)
                txt_path = os.path.join(lbl_dir, os.path.splitext(img_name)[0] + ".txt")
                
                with open(txt_path, 'w') as f:
                    if cid != 3:
                        xc, yc, w, h = get_smart_bbox(img)
                        f.write(f"{cid} {xc:.4f} {yc:.4f} {w:.4f} {h:.4f}\n")

if __name__ == "__main__":
    run_fix()
    print("\n✅ All labels have been fixed and synced with your images!")