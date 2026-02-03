import torch
from torch.utils.data import DataLoader
from dataset import SegmentationDataset
from unet_model import UNet

# -------------------------
# Dataset paths (MATCH YOUR STRUCTURE)
# -------------------------
TRAIN_IMG = "model/segmentation/segmentation_dataset/images/train"
TRAIN_MASK = "model/segmentation/segmentation_dataset/masks/train"

VAL_IMG = "model/segmentation/segmentation_dataset/images/val"
VAL_MASK = "model/segmentation/segmentation_dataset/masks/val"

# -------------------------
# Load Dataset
# -------------------------
train_dataset = SegmentationDataset(TRAIN_IMG, TRAIN_MASK)
val_dataset = SegmentationDataset(VAL_IMG, VAL_MASK)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)

# -------------------------
# Model Setup
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.BCELoss()

# -------------------------
# Training Loop
# -------------------------
epochs = 10

for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        preds = model(images)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f}")

# -------------------------
# Save Model
# -------------------------
torch.save(model.state_dict(), "model/segmentation/unet_best.pth")
print("✅ U-Net training completed and model saved")
