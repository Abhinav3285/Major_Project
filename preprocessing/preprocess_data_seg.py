import cv2
import os
import numpy as np
import random
import shutil
from tqdm import tqdm

def get_project_root():
    current_path = os.path.abspath(__file__)
    while os.path.basename(current_path) != "NeuroScanVision":
        parent = os.path.dirname(current_path)
        if parent == current_path: return os.path.dirname(os.path.abspath(__file__))
        current_path = parent
    return current_path

BASE_DIR = get_project_root()
RAW_DIR = os.path.join(BASE_DIR, "raw_data")
DEST_DIR = os.path.join(BASE_DIR, "data", "yolo_segmentation")

def get_segmentation_polygon(img):
    """Traces the tumor shape and returns normalized coordinates."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_cnt = max(contours, key=cv2.contourArea)
        # Simplify the shape to keep the file small
        epsilon = 0.002 * cv2.arcLength(largest_cnt, True)
        approx = cv2.approxPolyDP(largest_cnt, epsilon, True)
        
        h, w = img.shape[:2]
        points = []
        for p in approx:
            points.append(f"{p[0][0]/w:.4f} {p[0][1]/h:.4f}")
        return " ".join(points)
    return None

def setup_folders():
    if os.path.exists(DEST_DIR): shutil.rmtree(DEST_DIR)
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(DEST_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(DEST_DIR, "labels", split), exist_ok=True)

def run():
    print(f"📐 Creating Segmentation Dataset in: {DEST_DIR}")
    setup_folders()
    categories = [("glomia", 0), ("meningioma", 1), ("pituitary", 2), ("no_tumor", 3)]
    
    for cat, cid in categories:
        src = os.path.join(RAW_DIR, cat)
        if not os.path.exists(src): continue
        imgs = [f for f in os.listdir(src) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        random.shuffle(imgs)
        
        # Split 70/20/10
        t_end, v_end = int(len(imgs)*0.7), int(len(imgs)*0.9)
        splits = {"train": imgs[:t_end], "val": imgs[t_end:v_end], "test": imgs[v_end:]}

        for split_name, img_list in splits.items():
            for img_name in tqdm(img_list, desc=f"{cat} -> {split_name}"):
                img = cv2.imread(os.path.join(src, img_name))
                if img is None: continue
                img_res = cv2.resize(img, (640, 640))
                
                # Save Image
                unique_name = f"{cat}_{img_name}"
                cv2.imwrite(os.path.join(DEST_DIR, "images", split_name, unique_name), img_res)

                # Save Segmentation Label
                polygon = get_segmentation_polygon(img_res)
                if polygon and cat != "no_tumor":
                    with open(os.path.join(DEST_DIR, "labels", split_name, os.path.splitext(unique_name)[0]+".txt"), 'w') as f:
                        f.write(f"{cid} {polygon}\n")

if __name__ == "__main__":
    run()