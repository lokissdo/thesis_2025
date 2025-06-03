import os
import glob
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from eval_utils.oracle_celebahq_metrics import OracleResnet

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

def load_img(path, device):
    if not os.path.exists(path):
        print(f"Warning ---- File not found: {path}")
        return None
    try:
        img = Image.open(path).convert("RGB")
        img = preprocess(img)
        return img.to(device, dtype=torch.float).unsqueeze(0)
    except Exception as e:
        print(f"Error ---- Failed to load image {path}: {e}")
        return None

def find_matching_image(directory, num_image):
    possible_files = glob.glob(os.path.join(directory, f"{num_image}.*"))
    return possible_files[0] if possible_files else None

@torch.no_grad()
def get_attrs_and_target(actual_dir, target_dir, oracle, device):
    oracle_preds = {'cf': {'dist': [], 'pred': []}, 'cl': {'dist': [], 'pred': []}}
    actual_files = sorted(os.listdir(actual_dir))
    processed_count = 0

    for file in tqdm(actual_files, desc="Processing Images"):
        num_image = file.split('.')[0]
        actual_file = find_matching_image(actual_dir, num_image)
        target_file = find_matching_image(target_dir, num_image)

        cl = load_img(actual_file, device)
        cf = load_img(target_file, device)

        if cl is None or cf is None:
            continue

        cl_o_dist = oracle(cl)
        cf_o_dist = oracle(cf)

        oracle_preds['cl']['dist'].append(cl_o_dist.cpu().numpy())
        oracle_preds['cl']['pred'].append((cl_o_dist > 0.5).cpu().numpy())
        oracle_preds['cf']['dist'].append(cf_o_dist.cpu().numpy())
        oracle_preds['cf']['pred'].append((cf_o_dist > 0.5).cpu().numpy())

        processed_count += 1

    if processed_count == 0:
        print("No images were processed successfully. Please check the input directories.")
        return None

    for key in ['cl', 'cf']:
        for sub_key in ['dist', 'pred']:
            if oracle_preds[key][sub_key]:
                oracle_preds[key][sub_key] = np.concatenate(oracle_preds[key][sub_key])
            else:
                oracle_preds[key][sub_key] = np.array([])

    return oracle_preds

def compute_CorrMetric(actual_dir, target_dir, oracle, device, query_label, diff=True, remove_unchanged_oracle=False):
    oracle_preds = get_attrs_and_target(actual_dir, target_dir, oracle, device)
    if oracle_preds is None:
        return None

    cf_pred = oracle_preds['cf']['pred'].astype('float')
    cl_pred = oracle_preds['cl']['pred'].astype('float')

    delta_query = cf_pred[:, query_label] - cl_pred[:, query_label] if diff else cf_pred[:, query_label]
    deltas = cf_pred - cl_pred if diff else cf_pred

    if remove_unchanged_oracle:
        to_remove = cf_pred[:, query_label] != cl_pred[:, query_label]
        deltas = deltas[to_remove, :]
        delta_query = delta_query[to_remove]

    print('Dataset Size After Filtering:', len(deltas))

    our_corrs = np.zeros(40)
    for i in range(40):
        cc = np.corrcoef(deltas[:, i], delta_query)
        our_corrs[i] = 0 if np.any(np.isnan(cc)) else cc[0, 1]

    return our_corrs

class CelebaHQOracle():
    def __init__(self, weights_path, device):
        oracle = OracleResnet(weights_path=None, freeze_layers=True)
        oracle.load_state_dict(torch.load(weights_path, map_location='cpu')['model_state_dict'])
        self.oracle = oracle.to(device)
        self.oracle.eval()

    def __call__(self, x):
        return self.oracle(x)

def parse_args():
    parser = argparse.ArgumentParser(description="Compute CD Metric on CelebA-HQ")
    parser.add_argument('--oracle-path', type=str, required=True, help="Path to oracle model checkpoint")
    parser.add_argument('--actual-path', type=str, required=True, help="Path to original images")
    parser.add_argument('--output-path', type=str, required=True, help="Path to counterfactual images")
    parser.add_argument('--celeba-path', type=str, required=True, help="Path to CelebA-HQ root directory")
    parser.add_argument('--dataset', type=str, default="CelebAHQ", help="Dataset name")
    parser.add_argument('--query-label', type=int, required=True, help="Query label index")
    parser.add_argument('--gpu', type=int, default=0, help="GPU index to use")
    return parser.parse_args()

def main():
    args = parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    actual_dir = args.actual_path
    target_dir = args.output_path
    weights_path = args.oracle_path
    query_label = args.query_label

    print("=" * 80)
    print(f"Computing CD Metric on: {args.dataset}")
    print(f"Original images: {actual_dir}")
    print(f"Counterfactual images: {target_dir}")
    print(f"Query Label: {query_label}")
    print(f"Using oracle checkpoint: {weights_path}")
    print("=" * 80)

    oracle = CelebaHQOracle(weights_path=weights_path, device=device)

    results = compute_CorrMetric(actual_dir,
                                 target_dir,
                                 oracle,
                                 device,
                                 query_label,
                                 diff=True,
                                 remove_unchanged_oracle=False)

    if results is not None:
        print("CD Result:", np.sum(np.abs(results)))

if __name__ == "__main__":
    main()