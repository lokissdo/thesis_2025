import os
import glob
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils import data
from torchvision import models, transforms
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Compute Flip Ratio between original and adversarial images.")
    parser.add_argument("--actual-dir", type=str, required=True, help="Directory of original images")
    parser.add_argument("--target-dir", type=str, required=True, help="Directory of adversarial images")
    parser.add_argument("--weights", type=str, required=True, help="Path to classifier model weights (.pth)")
    parser.add_argument("--model", type=str, default="resnet50", help="Classifier model architecture")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for DataLoader")
    parser.add_argument("--query-label", type=int, default=31, help="Target label index to evaluate (e.g., Smile = 31)")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save result text files")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to use (default: 0)")
    return parser.parse_args()


class MultiLabelModel(nn.Module):
    def __init__(self, model_name='resnet50', num_labels=40):
        super(MultiLabelModel, self).__init__()

        if model_name == 'resnet50':
            self.base_model = models.resnet50(pretrained=True)
            num_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()

            self.fc1 = nn.Linear(num_features, 1024)
            self.bn1 = nn.BatchNorm1d(1024)
            self.relu1 = nn.ReLU()
            self.drop1 = nn.Dropout(0.5)

            self.fc2 = nn.Linear(1024, 512)
            self.bn2 = nn.BatchNorm1d(512)
            self.relu2 = nn.ReLU()

            self.fc3 = nn.Linear(512, num_labels)
        else:
            raise ValueError(f"Model '{model_name}' is not supported.")

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.base_model(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


def load_classifier(weights_path, model_name, device):
    model = MultiLabelModel(model_name=model_name, num_labels=40).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class CFDataset(data.Dataset):
    def __init__(self, actual_dir, target_dir):
        self.actual_images = sorted(glob.glob(os.path.join(actual_dir, "*.png")))
        self.target_dir = target_dir

    def __len__(self):
        return len(self.actual_images)

    def __getitem__(self, idx):
        actual_path = self.actual_images[idx]
        img_name = os.path.basename(actual_path).replace(".png", "")
        target_path = os.path.join(self.target_dir, f"{img_name}.png")

        if not os.path.exists(target_path):
            return None, None, img_name

        actual_img = self.load_img(actual_path)
        target_img = self.load_img(target_path)
        return actual_img, target_img, img_name

    def load_img(self, path):
        try:
            img = Image.open(path).convert("RGB")
            return transform(img)
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            return None


def compute_flip_ratio(classifier, loader, device, query_label):
    total_samples = 0
    flipped_predictions = 0
    flipped_0_to_1 = []
    flipped_1_to_0 = []

    with torch.no_grad():
        for cl_batch, cf_batch, names in tqdm(loader, desc="Computing Flip Ratio"):
            valid_indices = [i for i in range(len(cl_batch)) if cl_batch[i] is not None and cf_batch[i] is not None]
            if not valid_indices:
                continue

            cl_tensor = torch.stack([cl_batch[i] for i in valid_indices]).to(device)
            cf_tensor = torch.stack([cf_batch[i] for i in valid_indices]).to(device)
            valid_names = [names[i] for i in valid_indices]

            pred_cl = classifier(cl_tensor)[:, query_label] > 0.5
            pred_cf = classifier(cf_tensor)[:, query_label] > 0.5

            for i in range(len(valid_names)):
                if pred_cl[i] != pred_cf[i]:
                    flipped_predictions += 1
                    name = valid_names[i] + ".png"
                    if pred_cl[i] == 0 and pred_cf[i] == 1:
                        flipped_0_to_1.append(name)
                    else:
                        flipped_1_to_0.append(name)

            total_samples += len(valid_names)

    flip_ratio = flipped_predictions / total_samples if total_samples > 0 else 0
    return flip_ratio, flipped_0_to_1, flipped_1_to_0


def run_evaluation(args):
    print("=" * 80)
    print(f"Evaluating Flip Ratio for label {args.query_label}")
    print(f"Original images from: {args.actual_dir}")
    print(f"Counterfactual images from: {args.target_dir}")
    print(f"Model: {args.model} | Weights: {args.weights}")
    print("=" * 80)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    classifier = load_classifier(args.weights, args.model, device)
    dataset = CFDataset(args.actual_dir, args.target_dir)
    loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    flip_ratio, flipped_0_to_1, flipped_1_to_0 = compute_flip_ratio(classifier, loader, device, args.query_label)

    print(f"\nFlip Ratio for label {args.query_label}: {flip_ratio:.4f}")

    print("\nFiles flipped from 0 to 1:")
    for fname in flipped_0_to_1:
        print("   -", fname)

    print("\nFiles flipped from 1 to 0:")
    for fname in flipped_1_to_0:
        print("   -", fname)

    # Save to text files
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "flipped_0_to_1.txt"), "w") as f:
        for fname in flipped_0_to_1:
            f.write(fname + "\n")

    with open(os.path.join(args.output_dir, "flipped_1_to_0.txt"), "w") as f:
        for fname in flipped_1_to_0:
            f.write(fname + "\n")


if __name__ == "__main__":
    args = parse_arguments()
    run_evaluation(args)