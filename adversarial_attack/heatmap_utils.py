import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2

CAM_TYPES = {
    "ScoreCAM": ScoreCAM,
    "GradCAM": GradCAM,
    "GradCAMPlusPlus": GradCAMPlusPlus,
    "AblationCAM": AblationCAM,
    "XGradCAM": XGradCAM,
    "EigenCAM": EigenCAM,
    "FullGrad": FullGrad,
}

def generate_mask(heatmap, epsilon=0.5):
    return (heatmap >= epsilon).astype(np.uint8)

def unnormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1, 3, 1, 1)
    return tensor * std + mean

def generate_heatmap_and_mask(image_path, model, target_layer, label_index, cam_type="GradCAM", threshold=0.5, input_size=512, device='cuda'):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)

    if cam_type not in CAM_TYPES:
        raise ValueError(f"Unsupported CAM type: {cam_type}")

    cam = CAM_TYPES[cam_type](model=model, target_layers=[target_layer])

    def target_specific_label(model_output):
        if model_output.dim() == 1:
            model_output = model_output.unsqueeze(0)
        return model_output[:, label_index]

    grayscale_cam = cam(input_tensor=image_tensor, targets=[target_specific_label])[0].cpu().numpy()
    rgb_image = np.array(image.resize((input_size, input_size))) / 255.0
    heatmap = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

    mask_np = generate_mask(grayscale_cam, epsilon=threshold)
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).float().to(device)
    mask_tensor = mask_tensor.repeat(1, 3, 1, 1)

    return image_tensor, grayscale_cam, mask_tensor, heatmap, image