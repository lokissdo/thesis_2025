import os
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import torch.nn as nn

# ------------------ SimSiam Model ------------------
class SimSiam(nn.Module):
    def __init__(self, base_encoder, dim=2048, pred_dim=512):
        super(SimSiam, self).__init__()
        self.criterion = nn.CosineSimilarity(dim=1)
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)

        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),
            self.encoder.fc,
            nn.BatchNorm1d(dim, affine=False)
        )
        self.encoder.fc[6].bias.requires_grad = False
        self.predictor = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, dim)
        )

    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        dist = self.criterion(z1, z2)
        return dist

# ------------------ Load Pretrained ------------------
def get_simsiam_dist(weights_path):
    import torchvision.models as models
    model = SimSiam(models.resnet50, dim=2048, pred_dim=512)
    state_dict = torch.load(weights_path, map_location='cpu')['state_dict']
    model.load_state_dict({k[7:]: v for k, v in state_dict.items()})
    return model

# ------------------ Preprocessing ------------------
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def load_img(path, device):
    img = Image.open(path).convert("RGB")
    img = preprocess(img)
    return img.unsqueeze(0).to(device, dtype=torch.float)

# ------------------ Compute Similarity ------------------
def compute_FVA(oracle, actual_dir, target_dir, device):
    dists = []
    cnt = 0
    with torch.no_grad():
        for i in range(300):
            input_path = os.path.join(actual_dir, f'sample_{i}.png')
            output_path = os.path.join(target_dir, f'sample_{i}.png')

            if not os.path.exists(input_path) or not os.path.exists(output_path):
                print(f"Missing file: {input_path} or {output_path}")
                continue

            input_tensor = load_img(input_path, device)
            output_tensor = load_img(output_path, device)
            dist = oracle(input_tensor, output_tensor).cpu().numpy()
            dists.append(dist)
            cnt += 1

    print("Processed count:", cnt)
    return np.concatenate(dists)

# ------------------ Argparse ------------------
def parse_args():
    parser = argparse.ArgumentParser(description='SimSiam Similarity Arguments')
    parser.add_argument('--weights-path', type=str, required=True,
                        help='Path to pretrained SimSiam ResNet50 weights')
    parser.add_argument('--actual-dir', type=str, required=True,
                        help='Path to original images')
    parser.add_argument('--target-dir', type=str, required=True,
                        help='Path to counterfactual images')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU index (default: 0)')
    return parser.parse_args()

# ------------------ Main ------------------
if __name__ == '__main__':
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    oracle = get_simsiam_dist(args.weights_path)
    oracle.to(device)
    oracle.eval()

    results = compute_FVA(oracle, args.actual_dir, args.target_dir, device)
    print('SimSiam Cosine Similarity (mean): {:.4f}'.format(np.mean(results).item()))