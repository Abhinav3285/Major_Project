import os

def create_labels_for_folder(img_folder, output_label_folder, class_id):
    if not os.path.exists(output_label_folder):
        os.makedirs(output_label_folder)
        
    for filename in os.listdir(img_folder):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            # Create the .txt filename
            txt_name = os.path.splitext(filename)[0] + ".txt"
            
            # This creates a box in the middle of the image (x_center, y_center, width, height)
            # 0.5 0.5 0.8 0.8 means a box covering 80% of the image
            with open(os.path.join(output_label_folder, txt_name), 'w') as f:
                f.write(f"{class_id} 0.5 0.5 0.7 0.7\n")
    
    print(f"✅ Generated labels for {img_folder}")

# Usage
create_labels_for_folder("raw_data/glomia", "data/yolo_dataset/labels/train", 0)
create_labels_for_folder("raw_data/meningioma", "data/yolo_dataset/labels/train", 1)
create_labels_for_folder("raw_data/pituitary", "data/yolo_dataset/labels/train", 2)
create_labels_for_folder("raw_data/no_tumor", "data/yolo_dataset/labels/train", 3)