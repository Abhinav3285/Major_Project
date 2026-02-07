import cv2
import os
import numpy as np
from tqdm import tqdm
import shutil

DATA_ROOT = "C:/Users/AbhinavKumar/Desktop/Project/NeuroScanVision/data/yolo_segmentation"

def mask_to_polygon(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None: return None
    # Threshold at 10 to catch even faint tumors
    _, thresh = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    h, w = mask.shape
    for cnt in contours:
        if cv2.contourArea(cnt) < 10: continue
        poly = (cnt.reshape(-1, 2) / np.array([w, h])).flatten()
        polygons.append(poly)
    return polygons

def get_class_id(filename):
    name = filename.lower()
    if "gl" in name: return 0      # Glioma
    if "me" in name: return 1      # Meningioma
    if "pi" in name: return 2      # Pituitary
    return 3                       # No Tumor

# Clear all old labels first to start fresh
if os.path.exists(os.path.join(DATA_ROOT, "labels")):
    shutil.rmtree(os.path.join(DATA_ROOT, "labels"))

for split in ["train", "val", "test"]:
    img_dir = os.path.join(DATA_ROOT, "images", split)
    mask_dir = os.path.join(DATA_ROOT, "masks", split)
    label_dir = os.path.join(DATA_ROOT, "labels", split)

    if not os.path.exists(img_dir): continue
    os.makedirs(label_dir, exist_ok=True)
    
    print(f"🔄 Syncing {split} split...")
    count = 0
    
    # Iterate through images
    for img_name in tqdm(os.listdir(img_dir)):
        base = os.path.splitext(img_name)[0] # e.g. 'Tr-gl_0014'
        
        # Construct the exact mask name we found earlier
        mask_name = f"{base}_m.jpg"
        mask_path = os.path.join(mask_dir, mask_name)
        
        if os.path.exists(mask_path):
            polys = mask_to_polygon(mask_path)
            if polys:
                cid = get_class_id(base)
                # Save label as 'Tr-gl_0014.txt' (Matches the Image Name!)
                with open(os.path.join(label_dir, f"{base}.txt"), 'w') as f:
                    for p in polys:
                        poly_str = " ".join([f"{c:.4f}" for c in p])
                        f.write(f"{cid} {poly_str}\n")
                count += 1

    print(f"✅ Success: Created {count} matching labels for {split}.")

print("\n🚀 DONE! Now run 'python model/train_seg.py'")