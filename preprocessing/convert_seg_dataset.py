import cv2
import os
import numpy as np
from tqdm import tqdm

# --- SET YOUR ROOT DATA PATH ---
DATA_ROOT = "C:/Users/AbhinavKumar/Desktop/Project/NeuroScanVision/data/yolo_segmentation"

def get_class_id(filename):
    """Infers the class ID from the filename."""
    name = filename.lower()
    if "glioma" in name: return 0
    if "meningioma" in name: return 1
    if "pituitary" in name: return 2
    return 3

def mask_to_polygon(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None: return None
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    h, w = mask.shape
    
    for cnt in contours:
        if cv2.contourArea(cnt) < 20: continue
        # Normalize and flatten
        poly = (cnt.reshape(-1, 2) / np.array([w, h])).flatten()
        polygons.append(poly)
    return polygons

# Process Train, Val, and Test
for split in ["train", "val", "test"]:
    mask_dir = os.path.join(DATA_ROOT, "masks", split)
    label_dir = os.path.join(DATA_ROOT, "labels", split)
    
    if not os.path.exists(mask_dir):
        print(f"⚠️ Skipping {split}: Folder not found")
        continue

    os.makedirs(label_dir, exist_ok=True)
    print(f"📂 Processing {split} masks...")

    for filename in tqdm(os.listdir(mask_dir)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            cid = get_class_id(filename)
            polys = mask_to_polygon(os.path.join(mask_dir, filename))
            
            if polys:
                txt_name = os.path.splitext(filename)[0] + ".txt"
                with open(os.path.join(label_dir, txt_name), 'w') as f:
                    for p in polys:
                        poly_str = " ".join([f"{coord:.4f}" for coord in p])
                        f.write(f"{cid} {poly_str}\n")

print("\n✅ DONE! Your 'labels' folder is now ready.")
print(f"📍 Location: {os.path.join(DATA_ROOT, 'labels')}")