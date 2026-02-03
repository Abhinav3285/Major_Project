from ultralytics import YOLO

def evaluate_model():
    # 1. Load your trained model
    model = YOLO('../results/brain_tumor_run/weights/best.pt')

    # 2. Run validation on the test set
    metrics = model.val(data='../data/data.yaml', split='test')

    # 3. Print Key Medical Metrics
    print(f"Mean Average Precision (mAP50): {metrics.box.map50:.2f}")
    print(f"Precision: {metrics.box.mp:.2f}")
    print(f"Recall: {metrics.box.mr:.2f}")
    
    # This automatically saves Confusion Matrix and PR Curves to the results folder

if __name__ == "__main__":
    evaluate_model()