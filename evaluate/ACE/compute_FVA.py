import os
import argparse
import torch
import numpy as np
from torch.utils import data
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from eval_utils.resnet50_facevgg2_FVA import resnet50, load_state_dict

# Dataset để load ảnh counterfactual
class CFDataset():
    mean_bgr = np.array([91.4953, 103.8827, 131.0912])

    def __init__(self, cf_folder):
        self.images = [img for img in os.listdir(cf_folder) if img.endswith(('.jpg', '.png'))]
        self.cf_folder = cf_folder

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        cf_path = os.path.join(self.cf_folder, img_name)
        cf = self.load_img(cf_path)
        return cf, img_name

    def load_img(self, path):
        img = Image.open(path).convert('RGB')
        img = transforms.Resize(224)(img)
        img = np.array(img, dtype=np.uint8)
        return self.transform(img)

    def transform(self, img):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float32)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        return torch.from_numpy(img).float()

# Hàm tính FVA
@torch.no_grad()
def compute_FVA(oracle, cf_folder, device, batch_size=15):
    dataset = CFDataset(cf_folder)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    cosine_similarity = torch.nn.CosineSimilarity()

    FVAS = []
    dists = []

    for cf, _ in tqdm(loader):
        cf = cf.to(device, dtype=torch.float)
        cf_feat = oracle(cf)
        dist = cosine_similarity(cf_feat, cf_feat)
        FVAS.append((dist > 0.5).cpu().numpy())
        dists.append(dist.cpu().numpy())

    return np.concatenate(FVAS), np.concatenate(dists)

# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Compute FVA using pretrained ResNet50")
    parser.add_argument('--cf-folder', type=str, required=True,
                        help='Path to folder containing counterfactual images')
    parser.add_argument('--weights-path', type=str, required=True,
                        help='Path to pretrained ResNet50 weights (pickle file)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU index (default: 0)')
    parser.add_argument('--batch-size', type=int, default=15,
                        help='Batch size (default: 15)')
    return parser.parse_args()

# Entry point
if __name__ == '__main__':
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    oracle = resnet50(num_classes=8631, include_top=False).to(device)
    load_state_dict(oracle, args.weights_path)
    oracle.eval()

    results = compute_FVA(oracle, args.cf_folder, device, args.batch_size)

    print('FVA (Proportion):', np.mean(results[0]))
    print('FVA (STD):', np.std(results[0]))
    print('Mean Cosine Distance:', np.mean(results[1]))
    print('STD Cosine Distance :', np.std(results[1]))