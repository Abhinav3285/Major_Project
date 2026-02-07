import cv2
import os
import numpy as np
import random
import shutil
from tqdm import tqdm

# --- DYNAMIC PROJECT ROOT FINDER ---
def get_project_root():
    current_path = os.path.abspath(__file__)
    while os.path.basename(current_path) != "NeuroScanVision":
        parent = os.path.dirname(current_path)
        if parent == current_path: # Root reached
            return os.path.dirname(os.path.abspath(__file__))
        current_path = parent
    return current_path

BASE_DIR = get_project_root()
RAW_DIR = os.path.join(BASE_DIR, "raw_data")
DEST_DIR = os.path.join(BASE_DIR, "data", "yolo_dataset")

def get_smart_bbox(img):
    """Detects brightest area (tumor) and returns YOLO format coords."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_cnt)
        img_h, img_w = img.shape[:2]
        return (x + w/2) / img_w, (y + h/2) / img_h, w / img_w, h / img_h
    return 0.5, 0.5, 0.15, 0.15

def setup_folders():
    if os.path.exists(DEST_DIR):
        shutil.rmtree(DEST_DIR)
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(DEST_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(DEST_DIR, "labels", split), exist_ok=True)

def run_preprocessing():
    print(f"🚀 Starting Preprocessing in: {BASE_DIR}")
    setup_folders()
    
    categories = [("glomia", 0), ("meningioma", 1), ("pituitary", 2), ("no_tumor", 3)]
    
    for cat_name, class_id in categories:
        input_folder = os.path.join(RAW_DIR, cat_name)
        if not os.path.exists(input_folder):
            print(f"⚠️ Skipping {cat_name}: Folder not found")
            continue

        images = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        random.shuffle(images)
        
        # Split: 70% Train, 20% Val, 10% Test
        train_end = int(len(images) * 0.7)
        val_end = int(len(images) * 0.9)
        splits = {"train": images[:train_end], "val": images[train_end:val_end], "test": images[val_end:]}

        for split_name, img_list in splits.items():
            img_out = os.path.join(DEST_DIR, "images", split_name)
            lbl_out = os.path.join(DEST_DIR, "labels", split_name)

            for img_name in tqdm(img_list, desc=f"Processing {cat_name} ({split_name})"):
                img = cv2.imread(os.path.join(input_folder, img_name))
                if img is None: continue
                img_res = cv2.resize(img, (640, 640))
                
                new_name = f"{cat_name}_{img_name}"
                cv2.imwrite(os.path.join(img_out, new_name), img_res)

                txt_name = os.path.splitext(new_name)[0] + ".txt"
                with open(os.path.join(lbl_out, txt_name), 'w') as f:
                    if cat_name != "no_tumor":
                        xc, yc, w, h = get_smart_bbox(img_res)
                        f.write(f"{class_id} {xc:.4f} {yc:.4f} {w:.4f} {h:.4f}\n")

if __name__ == "__main__":
    run_preprocessing()
    print("\n✅ Dataset Ready at: data/yolo_dataset/")