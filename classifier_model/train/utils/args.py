import argparse
import torch

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a multi-label classification model on CelebA dataset.")
    parser.add_argument('--root_dir', type=str, required=True, help='Path to the CelebA images directory.')
    parser.add_argument('--attr_file', type=str, required=True, help='Path to the attribute file.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return args