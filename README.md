# 🧠 Brain Tumor Detection using YOLOv8

This project uses the **YOLOv8** (You Only Look Once) model to directly find and identify brain tumors in MRI scans. Unlike older systems that just say "yes" or "no" to a tumor, this system draws a box exactly where the tumor is located.

## 🌟 Key Features
* **Real-time Detection:** Finds tumors almost instantly.
* **Multi-Class:** Can tell the difference between Glioma, Meningioma, and Pituitary tumors.
* **Web Interface:** Easy-to-use website built with Streamlit.

## 📂 Project Structure
* `data/`: Contains the MRI images and the "Answer Key" labels.
* `model/`: The brain of the project where we train the YOLOv8 model.
* `app/`: The code for the website where you upload images.
* `preprocessing/`: Scripts to clean and organize the photos.

## 🛠️ How to Set It Up

### 1. Install the tools
Open your terminal and run:
```bash
pip install ultralytics streamlit opencv-python numpy

2. Prepare the Data
Ensure your images are in the data/yolo_dataset/images folder and your labels are in data/yolo_dataset/labels.

3. Train the Model
To start the computer's study session, run:

Bash

cd model
python train_yolo.py
4. Run the Website
To launch your brain tumor detection app, run:

Bash

cd app
streamlit run app.py
📊 Results
Once trained, the model provides:

mAP (Mean Average Precision): How accurate the boxes are.

Confusion Matrix: Shows if the model is confusing one tumor for another.

🎓 Acknowledgments
Dataset provided by Roboflow.

Model architecture by Ultralytics.