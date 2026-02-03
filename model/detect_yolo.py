import os
import cv2
from ultralytics import YOLO

# 1. SETUP PATHS (Absolute paths to prevent "File Not Found" errors)
BASE_DIR = r"E:\Brain-Tumor-Detection-YOLO"
MODEL_PATH = os.path.join(BASE_DIR, "results", "brain_tumor_run", "weights", "best.pt")
SOURCE_PATH = os.path.join(BASE_DIR, "data", "raw") # Folder with your MRI images

# 2. LOAD MODEL
if not os.path.exists(MODEL_PATH):
    print(f"❌ ERROR: Model not found at {MODEL_PATH}")
    exit()

model = YOLO(MODEL_PATH)

def run_detection():
    print(f"🚀 Starting detection on images in: {SOURCE_PATH}")
    
    # 3. RUN PREDICT & SAVE TO VISUALIZATION FOLDER
    # project='results' and name='visualization' tells YOLO exactly where to put the output
    results = model.predict(
        source=SOURCE_PATH,
        conf=0.5,           # 50% confidence threshold
        save=True,          # Save images with boxes
        project=os.path.join(BASE_DIR, 'results'), 
        name='visualization', 
        exist_ok=True       # Overwrite if folder exists instead of creating 'visualization2'
    )

    # 4. SHOW RESULTS ON SCREEN
    for result in results:
        # result.plot() returns the image with boxes as a numpy array (BGR for OpenCV)
        annotated_frame = result.plot()
        
        # Display the window
        cv2.imshow("YOLOv8 Brain Tumor Detection", annotated_frame)
        
        # Wait for 1.5 seconds per image or press 'q' to stop
        if cv2.waitKey(1500) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print(f"✅ Detection finished. Check your '{BASE_DIR}\\results\\visualization' folder!")

if __name__ == "__main__":
    run_detection()