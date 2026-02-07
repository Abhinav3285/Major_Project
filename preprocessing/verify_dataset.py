import os

DATA_ROOT = "C:/Users/AbhinavKumar/Desktop/Project/NeuroScanVision/data/yolo_segmentation"

def verify_split(split_name):
    img_dir = os.path.join(DATA_ROOT, "images", split_name)
    lbl_dir = os.path.join(DATA_ROOT, "labels", split_name)

    if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
        print(f"⚠️  Split '{split_name}' folders missing.")
        return

    images = {os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))}
    labels = {os.path.splitext(f)[0] for f in os.listdir(lbl_dir) if f.endswith('.txt')}

    print(f"\n--- Checking {split_name.upper()} split ---")
    print(f"📸 Total Images: {len(images)}")
    print(f"📄 Total Labels: {len(labels)}")

    # 1. Check for images without labels
    missing_labels = images - labels
    if missing_labels:
        print(f"❌ ERROR: {len(missing_labels)} images are missing labels!")
        print(f"Example missing: {list(missing_labels)[:20]}")
    else:
        print("✅ All images have matching labels.")

    # 2. Check for empty label files
    empty_count = 0
    for lbl in labels:
        path = os.path.join(lbl_dir, lbl + ".txt")
        if os.path.getsize(path) == 0:
            empty_count += 1
    
    if empty_count > 0:
        print(f"⚠️  WARNING: Found {empty_count} EMPTY label files (0 bytes).")
    else:
        print("✅ No empty label files found.")

# Run the check
verify_split("train")
verify_split("val")
verify_split("test")

print("\n🚀 If you see 3 green checks for TRAIN, you are ready to train!")