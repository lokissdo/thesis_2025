import sys
sys.path.append('../blended-latent-diffusion')

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset
import torch.nn.functional as F
import pandas as pd
import cv2
import os
import numpy as np
from PIL import Image
import argparse

# ===================== Dataset =====================
class CelebADataset(Dataset):
    def __init__(self, root_dir, attr_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.attrs = pd.read_csv(attr_file, delim_whitespace=True, skiprows=1).iloc[:, :40].replace(-1, 0)
        self.images = self.attrs.index.tolist()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        labels = self.attrs.iloc[idx].values.astype(np.float32)
        return image, labels

# ===================== Model =====================
class MultiLabelModel(nn.Module):
    def __init__(self, model_name='resnet18', num_labels=40):
        super(MultiLabelModel, self).__init__()
        
        # Khởi tạo size dựa trên model_name
        if model_name == 'resnet18':
            self.size = 1024
            self.base_model = models.resnet18(pretrained=True)
            num_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Linear(num_features, num_labels)
        elif model_name in ['densenet121', 'vgg19']:
            self.size = 256
            if model_name == 'densenet121':
                self.base_model = models.densenet121(pretrained=True)
                num_features = self.base_model.classifier.in_features
                self.base_model.classifier = nn.Linear(num_features, num_labels)
            else:  # model_name == 'vgg19'
                self.base_model = models.vgg19(pretrained=True)
                num_features = self.base_model.classifier[6].in_features
                self.base_model.classifier[6] = nn.Linear(num_features, num_labels)
        elif model_name in ['resnet50', 'efficientNet']:
            self.size = 512
            if model_name == 'resnet50':
                self.base_model = models.resnet50(pretrained=True)
                num_features = self.base_model.fc.in_features
                # self.base_model.fc = nn.Linear(num_features, num_labels)
                self.base_model.fc = nn.Identity()  # Remove the original fully connected layer

                # Additional layers after the base model
                self.fc1 = nn.Linear(num_features, 1024)  # Add a new fully connected layer
                self.bn1 = nn.BatchNorm1d(1024)  # Batch normalization layer
                self.relu1 = nn.ReLU()  # ReLU activation
                self.drop1 = nn.Dropout(0.5)  # Dropout layer to prevent overfitting
        
                self.fc2 = nn.Linear(1024, 512)  # Another fully connected layer
                self.bn2 = nn.BatchNorm1d(512)  # Batch normalization
                self.relu2 = nn.ReLU()  # ReLU activation
        
                self.fc3 = nn.Linear(512, num_labels)  # Final layer for multi-label classification

            else:  # model_name == 'efficientnet'
                self.base_model = models.efficientnet_b0(pretrained=True)
                num_features = self.base_model.classifier[-1].in_features
                self.base_model.classifier[-1] = nn.Linear(num_features, num_labels)
        else:
            raise ValueError(f"Model '{model_name}' is not supported.")

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x = self.base_model(x)
        # x = self.sigmoid(x)  # Apply sigmoid activation for multi-label classification
        # return x
        x = self.base_model(x)  # Pass through ResNet50
        x = self.fc1(x)  # First fully connected layer
        x = self.bn1(x)  # Batch normalization
        x = self.relu1(x)  # ReLU activation
        x = self.drop1(x)  # Dropout layer

        x = self.fc2(x)  # Second fully connected layer
        x = self.bn2(x)  # Batch normalization
        x = self.relu2(x)  # ReLU activation

        x = self.fc3(x)  # Final layer
        x = self.sigmoid(x)  # Apply sigmoid activation

        return x        
    
# ===================== Predict =====================
def predict_attr_for_image(image_path, model, dataset, attr_index, transform):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(next(model.parameters()).device)

    image_name = os.path.basename(image_path)
    idx = dataset.images.index(image_name)
    groundtruth_label = dataset.attrs.iloc[idx, attr_index]

    with torch.no_grad():
        output = model(image_tensor).cpu().numpy()[0]
        prediction = int(output[attr_index] > 0.5)

    attr_name = dataset.attrs.columns[attr_index]
    return {
        "attribute_name": attr_name,
        "ground_truth": int(groundtruth_label),
        "model_prediction": prediction,
        "confidence_score": output[attr_index]
    }

# ===================== Argument Parser =====================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, required=True, help='Directory of CelebA images')
    parser.add_argument('--attr_file', type=str, required=True, help='Path to attribute annotation file')
    parser.add_argument('--mask_dir', type=str, required=True, help='Directory containing masks by index')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model weights')
    parser.add_argument('--out_dir', type=str, required=True, help='Root output directory')
    parser.add_argument('--attr_idx', type=int, default=31, help='Attribute index to predict')
    parser.add_argument('--start', type=int, default=0, help='Start index of images')
    parser.add_argument('--end', type=int, default=300, help='End index of images (exclusive)')
    return parser.parse_args()

# ===================== CLI Main =====================
if __name__ == '__main__':
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MultiLabelModel(model_name='resnet50', num_labels=40).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    dataset = CelebADataset(args.img_dir, args.attr_file)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]
    tf = transforms.Compose([
        transforms.Resize((model.size, model.size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.squeeze().tolist(), std=std.squeeze().tolist())
    ])

    for idx in range(args.start, args.end):
        img_path = os.path.join(args.img_dir, f'{idx}.jpg')
        outdir = os.path.join(args.out_dir, f'sample_{idx}')
        os.makedirs(outdir, exist_ok=True)

        res0 = predict_attr_for_image(img_path, model, dataset, args.attr_idx, transform=tf)
        print(res0)
        cur_pred = res0['model_prediction']

        x0 = tf(Image.open(img_path).convert('RGB')).unsqueeze(0).to(device)

        mask_face = cv2.imread(os.path.join(args.mask_dir, f'{idx}.jpg'), cv2.IMREAD_GRAYSCALE)
        _, _, h, w = x0.shape
        mask_resized = cv2.resize(mask_face, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_binary = cv2.threshold(
            cv2.dilate(mask_resized, np.ones((5, 5), np.uint8), 1), 1, 255, cv2.THRESH_BINARY
        )[1]
        mask_path = os.path.join(outdir, 'mask.png')
        cv2.imwrite(mask_path, mask_binary)

        if cur_pred == 0:
            prompt = "(photo-realistic:1.2), ultra-high-resolution portrait of the same person, keep eyes hair nose unchanged, add a natural warm smile with slightly up-turned lips and visible upper teeth, mouth perfectly centered in mask, soft cinematic lighting, shallow depth of field, 85 mm f/1.4, detailed skin texture, HDR"
        else:
            prompt = "(photo-realistic:1.2), ultra-high-resolution portrait of the same person, keep eyes hair nose unchanged, relax the mouth, lips closed, neutral facial expression, soft cinematic lighting, shallow depth of field, 85 mm f/1.4, detailed skin texture, HDR"

        command = f"""
        python3 ./bld/scripts/text_editing_SD2.py \
            --prompt \"{prompt}\" \
            --init_image \"{img_path}\" \
            --mask \"{mask_path}\" \
            --output_path \"{outdir}/res.jpg\" \
            --output_path_1 \"{outdir}/res_1.jpg\" \
            --output_path_2 \"{outdir}/res_2.jpg\" \
            --output_path_3 \"{outdir}/res_3.jpg\" \
            --output_path_4 \"{outdir}/res_4.jpg\"
        """
        os.system(command.strip())