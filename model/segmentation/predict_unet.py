# model/segmentation/predict_unet.py
import torch
import numpy as np
import cv2
from .unet_model import UNet   # ✅ RELATIVE IMPORT

MODEL_PATH = "model/segmentation/weights/unet_best.pth"

device = torch.device("cpu")

model = UNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

def predict_mask(image_np):
    img = cv2.resize(image_np, (256, 256))
    img = img / 255.0
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()

    with torch.no_grad():
        mask = model(img)[0][0].numpy()

    mask = (mask > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]))

    return mask
