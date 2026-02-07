import os

DATA_ROOT = "C:/Users/AbhinavKumar/Desktop/Project/NeuroScanVision/data/yolo_segmentation"

for split in ["train", "val", "test"]:
    img_dir = os.path.join(DATA_ROOT, "images", split)
    lbl_dir = os.path.join(DATA_ROOT, "labels", split)

    if not os.path.exists(img_dir): continue

    images = [os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    labels = [os.path.splitext(f)[0] for f in os.listdir(lbl_dir) if f.endswith('.txt')]

    missing = set(images) - set(labels)
    
    print(f"📁 Processing {split}: Creating {len(missing)} empty labels for background images...")
    
    for name in missing:
        # Create an empty .txt file
        with open(os.path.join(lbl_dir, name + ".txt"), 'w') as f:
            pass # Writes nothing, creates a 0-byte file

print("\n✅ Done! All healthy brain images now have empty label files.")