import os

LABEL_DIR = "C:/Users/AbhinavKumar/Desktop/Project/NeuroScanVision/data/yolo_segmentation/labels/train"

if not os.path.exists(LABEL_DIR):
    print(f"❌ ERROR: The folder does not exist: {LABEL_DIR}")
else:
    files = [f for f in os.listdir(LABEL_DIR) if f.endswith('.txt')]
    print(f"✅ Found {len(files)} files in the labels folder.")
    
    if len(files) > 0:
        # Check the first file
        first_file = os.path.join(LABEL_DIR, files[0])
        size = os.path.getsize(first_file)
        print(f"📄 First file: {files[0]}")
        print(f"⚖️ Size: {size} bytes")
        
        with open(first_file, 'r') as f:
            content = f.read().strip()
            if content:
                print(f"📝 Content start: {content[:50]}...")
            else:
                print("⚠️ WARNING: The file is EMPTY (0 characters).")