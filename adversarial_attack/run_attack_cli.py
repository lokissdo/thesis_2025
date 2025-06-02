import os
import argparse
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from model_utils import load_classifier
from pgd_utils import pgd_attack_smile_masked
from heatmap_utils import generate_heatmap_and_mask, unnormalize
import cv2

# ===== Argument Parser =====
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to original image')
    parser.add_argument('--adv_image', type=str, required=True, help='Path to adversarial input image')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save results')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model weights')
    parser.add_argument('--model_name', type=str, default='resnet50', help='Model name (default: resnet50)')
    parser.add_argument('--label_index', type=int, default=31, help='Label index to attack')
    parser.add_argument('--cam_type', type=str, default='GradCAM', help='CAM method')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for mask')
    parser.add_argument('--epsilon', type=float, default=0.2, help='PGD epsilon')
    parser.add_argument('--step_size', type=float, default=0.01, help='PGD step size')
    parser.add_argument('--nb_iter', type=int, default=500, help='Number of PGD iterations')
    return parser.parse_args()

# ===== Main Runner =====
if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_classifier(args.model_path, args.model_name, device)

    input_tensor, grayscale_cam, mask_tensor, heatmap, target_image = generate_heatmap_and_mask(
        args.image, model, target_layer=model.base_model.layer4[-1], label_index=args.label_index,
        cam_type=args.cam_type, threshold=args.threshold, input_size=model.size, device=device
    )

    with torch.no_grad():
        output_original = model(input_tensor)[0, args.label_index].item()
    print(f"Original score: {output_original:.4f}")

    transform = transforms.Compose([
        transforms.Resize((model.size, model.size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    adv_image = Image.open(args.adv_image).convert("RGB")
    x_adv_input = transform(adv_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output_adv = model(x_adv_input)[0, args.label_index].item()
    print(f"Adversarial input score: {output_adv:.4f}")

    target_label = 0.0 if output_original > 0.5 else 1.0
    y_target = torch.tensor([[target_label]], device=device)

    adv_x = pgd_attack_smile_masked(
        model, x_adv_input, y_target, mask=mask_tensor,
        epsilon=args.epsilon, step_size=args.step_size, nb_iter=args.nb_iter
    )

    with torch.no_grad():
        pred_before = model(x_adv_input)[0, args.label_index].item()
        pred_after = model(adv_x)[0, args.label_index].item()

    print(f"Before attack: {pred_before:.4f}, After attack: {pred_after:.4f}")

    original_img_np = unnormalize(input_tensor).squeeze(0).permute(1, 2, 0).cpu().numpy()
    adv_img_np = unnormalize(adv_x).squeeze(0).permute(1, 2, 0).cpu().numpy()

    plt.imsave(os.path.join(args.output_dir, 'original.png'), original_img_np)
    plt.imsave(os.path.join(args.output_dir, 'adversarial.png'), adv_img_np)
    cv2.imwrite(os.path.join(args.output_dir, 'mask.png'), (mask_tensor[0, 0].cpu().numpy() * 255).astype('uint8'))
    cv2.imwrite(os.path.join(args.output_dir, 'heatmap.png'), (heatmap * 255).astype('uint8'))

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(target_image)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap)
    plt.title("Heatmap")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(mask_tensor[0, 0].cpu().numpy(), cmap='gray')
    plt.title("Mask")
    plt.axis('off')

    plt.show()