import os

DATA_ROOT = "C:/Users/AbhinavKumar/Desktop/Project/NeuroScanVision/data/yolo_segmentation"

img_dir = os.path.join(DATA_ROOT, "images", "train")
mask_dir = os.path.join(DATA_ROOT, "labels", "train")

print("📸 FIRST 3 IMAGES:")
if os.path.exists(img_dir):
    print(os.listdir(img_dir)[:3])
else:
    print("Image folder not found!")

print("\n🎭 FIRST 3 MASKS:")
if os.path.exists(mask_dir):
    print(os.listdir(mask_dir)[:3])
else:
    print("Mask folder not found!")