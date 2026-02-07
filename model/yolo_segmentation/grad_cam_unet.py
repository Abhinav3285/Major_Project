import torch
import cv2
import numpy as np

def grad_cam_unet(model, image_tensor):
    feature_maps = []
    gradients = []

    def save_features(module, input, output):
        feature_maps.append(output)

    def save_grads(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    model.encoder4.register_forward_hook(save_features)
    model.encoder4.register_backward_hook(save_grads)

    output = model(image_tensor)
    output.mean().backward()

    fmap = feature_maps[0][0]
    grad = gradients[0][0]

    weights = grad.mean(dim=(1, 2))
    cam = torch.zeros(fmap.shape[1:], dtype=torch.float32)

    for i, w in enumerate(weights):
        cam += w * fmap[i]

    cam = cam.numpy()
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()

    cam = cv2.resize(cam, (256, 256))
    cam = (cam * 255).astype(np.uint8)

    return cam
