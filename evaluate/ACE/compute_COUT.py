import os
import torch
import argparse
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
from torchvision import transforms, models

# ======================= MultiLabelModel =======================
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
            raise ValueError(f"Unsupported model: {model_name}")

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

# ======================= Utility Functions =======================
def normalize_mask(deltas, mode='abs'):
    if mode == 'abs':
        deltas = torch.abs(deltas)
    elif mode == 'mse':
        deltas = deltas ** 2

    masks = torch.sum(deltas, dim=1)
    masks = masks.squeeze().view(deltas.shape[0], -1)
    masks -= masks.min(1, keepdim=True)[0]
    masks /= masks.max(1, keepdim=True)[0]
    return masks

def get_probs(x, model, actual_idx, target_idx):
    y = model(x.float())
    actual_prob, target_prob = y[:, actual_idx], y[:, target_idx]
    return actual_prob.view(-1, 1), target_prob.view(-1, 1)

def perturb_img(origin_img, target_img, indices=None):
    perturbed_img = origin_img.clone()
    if indices is not None:
        for batch in range(origin_img.shape[0]):
            perturbed_img[batch, :, indices[batch, :]] = target_img[batch, :, indices[batch, :]]
    return perturbed_img

def compute_aupc(probs):
    T = probs.shape[1] - 1
    sum_probs = torch.sum(probs[:, 1:-1], dim=1)
    aupc = (1 / (2 * T)) * (probs[:, 0] + 2 * sum_probs + probs[:, -1])
    return aupc

def compute_cout(origin_img, target_img, model, actual_idx, target_idx, device='cuda'):
    with torch.no_grad():
        origin_img = origin_img.to(device)
        target_img = target_img.to(device)

        x_f = origin_img.detach().clone()
        x_cf = target_img.detach().clone()
        deltas = x_f - x_cf

        masks = normalize_mask(deltas, mode='abs')
        sorted_indices = torch.argsort(masks, dim=1, descending=True)

        x_f = x_f.view(origin_img.shape[0], origin_img.shape[1], -1)
        x_cf = x_cf.view(target_img.shape[0], target_img.shape[1], -1)

        pixels_per_step = 200
        height, width = origin_img.shape[2], origin_img.shape[3]
        T = (height * width) // pixels_per_step

        actual_probs = []
        target_probs = []
        x_cur = perturb_img(x_f, x_cf, None)

        for step in range(T + 1):
            x_t = x_cur.view(origin_img.shape[0], origin_img.shape[1], height, width)
            actual_prob, target_prob = get_probs(x_t, model, actual_idx, target_idx)
            actual_probs.append(actual_prob)
            target_probs.append(target_prob)

            start = step * pixels_per_step
            end = min((step + 1) * pixels_per_step, x_f.shape[-1])
            if start < end:
                x_cur = perturb_img(x_cur, x_cf, sorted_indices[:, start:end])

        actual_probs = torch.cat(actual_probs, dim=1)
        target_probs = torch.cat(target_probs, dim=1)

        actual_aupc = compute_aupc(actual_probs)
        target_aupc = compute_aupc(target_probs)
        cout = target_aupc - actual_aupc
        return torch.sum(cout)

# ======================= CLI Interface =======================
def parse_arguments():
    parser = argparse.ArgumentParser(description="Compute COUT score between original and adversarial images.")
    parser.add_argument("--actual-dir", type=str, required=True, help="Path to original images")
    parser.add_argument("--target-dir", type=str, required=True, help="Path to adversarial images")
    parser.add_argument("--model-path", type=str, required=True, help="Path to classifier weights (.pth)")
    parser.add_argument("--model", type=str, default="resnet50", help="Model architecture (default: resnet50)")
    parser.add_argument("--actual-label", type=int, required=True, help="Actual label index")
    parser.add_argument("--target-label", type=int, required=True, help="Target label index")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    return parser.parse_args()

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_image_tensor(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
        return transform(img)
    except Exception as e:
        print(f"Could not load image {img_path}: {e}")
        return None

def main():
    args = parse_arguments()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    model = MultiLabelModel(model_name=args.model, num_labels=40).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    all_files = sorted(os.listdir(args.actual_dir))
    total_score = 0.0
    count = 0

    for filename in tqdm(all_files, desc="Computing COUT"):
        origin_path = os.path.join(args.actual_dir, filename)
        target_path = os.path.join(args.target_dir, filename)

        if not os.path.exists(origin_path) or not os.path.exists(target_path):
            continue

        origin_img = load_image_tensor(origin_path)
        target_img = load_image_tensor(target_path)

        if origin_img is None or target_img is None:
            continue

        origin_img = origin_img.unsqueeze(0).to(device)
        target_img = target_img.unsqueeze(0).to(device)

        try:
            score = compute_cout(origin_img, target_img, model, args.actual_label, args.target_label, device=device)
            total_score += score.item()
            count += 1
        except Exception as e:
            print(f"Error computing COUT for {filename}: {e}")

    if count == 0:
        print("No valid image pairs found.")
    else:
        average_score = total_score / count
        print(f"\nCOUT Average Score (from {count} samples): {average_score:.4f}")

if __name__ == "__main__":
    main()

# import os
# import torch
# import argparse
# import itertools
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import os.path as osp

# from PIL import Image
# from tqdm import tqdm
# from torch.utils import data
# from torchvision import models
# from torchvision import transforms

# from eval_utils.cout_metrics import evaluate

# from models.dive.densenet import DiVEDenseNet121
# from models.steex.DecisionDensenetModel import DecisionDensenetModel
# from models.mnist import Net

# from core.attacks_and_models import Normalizer

# from guided_diffusion.image_datasets import BINARYDATASET, MULTICLASSDATASETS


# # create dataset to read the counterfactual results images
# class CFDataset():
#     def __init__(self, path, exp_name):

#         self.images = []
#         self.path = path
#         self.exp_name = exp_name
#         for CL, CF in itertools.product(['CC'], ['CCF', 'ICF']):
#             self.images += [(CL, CF, I) for I in os.listdir(osp.join(path, 'Results', self.exp_name, CL, CF, 'CF'))]

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         CL, CF, I = self.images[idx]
#         # get paths
#         cl_path = osp.join(self.path, 'Original', 'Correct' if CL == 'CC' else 'Incorrect', I)
#         cf_path = osp.join(self.path, 'Results', self.exp_name, CL, CF, 'CF', I)

#         cl = self.load_img(cl_path)
#         cf = self.load_img(cf_path)

#         return cl, cf

#     def load_img(self, path):
#         img = Image.open(os.path.join(path))
#         img = np.array(img, dtype=np.uint8)
#         return self.transform(img)

#     def transform(self, img):
#         img = img.astype(np.float32) / 255
#         img = img.transpose(2, 0, 1)  # C x H x W
#         img = torch.from_numpy(img).float()
#         return img


# def arguments():
#     parser = argparse.ArgumentParser(description='COUT arguments.')
#     parser.add_argument('--gpu', default='0', type=str,
#                         help='GPU id')
#     parser.add_argument('--exp-name', required=True, type=str,
#                         help='Experiment Name')
#     parser.add_argument('--output-path', required=True, type=str,
#                         help='Results Path')
#     parser.add_argument('--weights-path', required=True, type=str,
#                         help='Classification model weights')
#     parser.add_argument('--dataset', required=True, type=str,
#                         choices=BINARYDATASET + MULTICLASSDATASETS,
#                         help='Dataset to evaluate')
#     parser.add_argument('--batch-size', default=10, type=int,
#                         help='Batch size')
#     parser.add_argument('--query-label', required=True, type=int)
#     parser.add_argument('--target-label', required=True, type=int)

#     return parser.parse_args()


# if __name__ == '__main__':
#     args = arguments()
#     device = torch.device('cuda:' + args.gpu)

#     dataset = CFDataset(args.output_path, args.exp_name)

#     loader = data.DataLoader(dataset, batch_size=args.batch_size,
#                              shuffle=False,
#                              num_workers=4, pin_memory=True)

#     print('Loading Classifier')

#     ql = args.query_label
#     if args.dataset in ['CelebA', 'CelebAMV']:
#         classifier = Normalizer(
#             DiVEDenseNet121(args.weights_path, args.query_label),
#             [0.5] * 3, [0.5] * 3
#         ).to(device)

#     elif args.dataset == 'CelebAHQ':
#         assert args.query_label in [20, 31, 39], 'Query label MUST be 20 (Gender), 31 (Smile), or 39 (Gender) for CelebAHQ'
#         ql = 0
#         if args.query_label in [31, 39]:
#             ql = 1 if args.query_label == 31 else 2
#         classifier = DecisionDensenetModel(3, pretrained=False,
#                                            query_label=ql)
#         classifier.load_state_dict(torch.load(args.weights_path, map_location='cpu')['model_state_dict'])
#         classifier = Normalizer(
#             classifier,
#             [0.5] * 3, [0.5] * 3
#         ).to(device)

#     elif 'BDD' in args.dataset:
#         classifier = DecisionDensenetModel(4, pretrained=False,
#                                            query_label=args.query_label)
#         classifier.load_state_dict(torch.load(args.weights_path, map_location='cpu')['model_state_dict'])
#         classifier = Normalizer(
#             classifier,
#             [0.5] * 3, [0.5] * 3
#         ).to(device)

#     else:
#         classifier = Normalizer(
#             models.resnet50(pretrained=True)
#         ).to(device)
    
#     classifier.eval()

#     results = evaluate(ql,
#                        args.target_label,
#                        classifier,
#                        loader,
#                        device,
#                        args.dataset in BINARYDATASET)
