import argparse
import os
from PIL import Image
import torch
from torchvision import transforms
from matplotlib import pyplot as plt

from model_utils import load_classifier
from pgd_utils import pgd_attack_smile_masked, unnormalize
def parse_arguments():
    parser = argparse.ArgumentParser(description="Adversarial attack on a single image.")
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--original', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--mask', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./output')
    return parser.parse_args()

def main():
    args = parse_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model = load_classifier(args.weights, 'resnet50', device)
    original_x = transform(Image.open(args.original).convert('RGB')).unsqueeze(0).to(device)
    input_x = transform(Image.open(args.input).convert('RGB')).unsqueeze(0).to(device)

    mask_img = Image.open(args.mask).convert("L").resize((512, 512))
    mask_tensor = transforms.ToTensor()(mask_img).to(device)
    mask_tensor = (mask_tensor > 0.5).float().unsqueeze(0).repeat(1, 3, 1, 1)

    with torch.no_grad():
        score_original = model(original_x)[0, 31].item()
        score_input = model(input_x)[0, 31].item()

    if round(score_input) != round(score_original):
        adv_x = input_x
        input_for_attack = input_x
    else:
        target_label = 0.0 if score_original > 0.5 else 1.0
        y_target = torch.tensor([[target_label]], device=device)
        input_for_attack = input_x if abs(target_label - score_input) < abs(target_label - score_original) else original_x
        adv_x = pgd_attack_smile_masked(model, input_for_attack, y_target, mask=mask_tensor, epsilon=0.3, step_size=0.5, nb_iter=500)

    with torch.no_grad():
        pred_before = model(original_x)[0, 31].item()
        pred_after = model(adv_x)[0, 31].item()

    print(f"Before attack: {pred_before:.4f}, After attack: {pred_after:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    for name, tensor in zip(['original', 'adversarial'], [original_x, adv_x]):
        img = unnormalize(tensor).squeeze(0).permute(1, 2, 0).cpu().clamp(0, 1).numpy()
        plt.imsave(os.path.join(args.output_dir, f'{name}.png'), img)

if __name__ == '__main__':
    main()