import cv2
import os
from tqdm import tqdm

def process_category(category_name, class_id):
    # Paths relative to this script (inside model/preprocessing)
    input_dir = f"../../raw_data/{category_name}"
    output_img_dir = "../../data/yolo_dataset/images/train"
    output_lbl_dir = "../../data/yolo_dataset/labels/train"

    if not os.path.exists(input_dir):
        print(f"⚠️ Folder not found, skipping: {input_dir}")
        return

    if not os.path.exists(output_img_dir): os.makedirs(output_img_dir)
    if not os.path.exists(output_lbl_dir): os.makedirs(output_lbl_dir)

    print(f"📦 Processing {category_name} (ID: {class_id})...")
    
    for img_name in tqdm(os.listdir(input_dir)):
        if img_name.endswith(('.jpg', '.png', '.jpeg')):
            # 1. Resize and Save Image
            img = cv2.imread(os.path.join(input_dir, img_name))
            if img is None: continue
            img_resized = cv2.resize(img, (640, 640))
            cv2.imwrite(os.path.join(output_img_dir, img_name), img_resized)

            # 2. Create matching YOLO Label file (.txt)
            txt_name = os.path.splitext(img_name)[0] + ".txt"
            with open(os.path.join(output_lbl_dir, txt_name), 'w') as f:
                # Format: class_id x_center y_center width height
                f.write(f"{class_id} 0.5 0.5 0.7 0.7\n")

if __name__ == "__main__":
    # This will process the images you put in raw_data and give them correct IDs
    process_category("glioma", 0)
    process_category("meningioma", 1)
    process_category("pituitary", 2)
    process_category("no_tumor", 3)
    print("\n✅ New images and labels added to data/yolo_dataset!")