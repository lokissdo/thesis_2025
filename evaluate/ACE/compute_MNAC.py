import os
import sys
sys.path.append("../")

import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from eval_utils.oracle_celebahq_metrics import OracleResnet

# Tiền xử lý ảnh
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def load_img(path, device):
    img = Image.open(path).convert('RGB')
    img = preprocess(img)
    return img.unsqueeze(0).to(device, dtype=torch.float)

def compute_MNAC(oracle, actual_dir, target_dir, device):
    MNACS = []
    dists = []
    cnt = 0
    with torch.no_grad():
        for i in range(300):
            input_path = os.path.join(actual_dir, f'sample_{i}.png')
            output_path = os.path.join(target_dir, f'sample_{i}.png')

            if not os.path.exists(input_path) or not os.path.exists(output_path):
                print(f"Skipping missing file: {input_path} or {output_path}")
                continue

            input_tensor = load_img(input_path, device)
            output_tensor = load_img(output_path, device)

            d_cl = oracle(input_tensor)
            d_cf = oracle(output_tensor)

            MNACS.append(((d_cl > 0.5) != (d_cf > 0.5)).sum(dim=1).cpu().numpy())
            dists.append([d_cl.cpu().numpy(), d_cf.cpu().numpy()])
            cnt += 1

    print("Processed count:", cnt)
    return np.concatenate(MNACS), np.concatenate([d[0] for d in dists]), np.concatenate([d[1] for d in dists])

class CelebaHQOracle():
    def __init__(self, weights_path, device):
        oracle = OracleResnet(weights_path=None, freeze_layers=True)
        oracle.load_state_dict(torch.load(weights_path, map_location='cpu')['model_state_dict'])
        self.oracle = oracle.to(device)
        self.oracle.eval()

    def __call__(self, x):
        return self.oracle(x)

def parse_args():
    parser = argparse.ArgumentParser(description='MNAC arguments.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to use')
    parser.add_argument('--oracle-path', type=str, required=True, help='Path to oracle checkpoint')
    parser.add_argument('--actual-dir', type=str, required=True, help='Path to original images')
    parser.add_argument('--target-dir', type=str, required=True, help='Path to counterfactual images')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    oracle = CelebaHQOracle(weights_path=args.oracle_path, device=device)

    results = compute_MNAC(oracle,
                           args.actual_dir,
                           args.target_dir,
                           device)

    print('MNAC Mean:', np.mean(results[0]))
    print('MNAC Std :', np.std(results[0]))