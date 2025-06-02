import sys
sys.path.append("../")
sys.path.append("../../")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from utils.args import parse_arguments
from utils.datasets import CelebADataset
from utils.train_utils import train_validate_loop

class ResNet50MultiLabel(nn.Module):
    def __init__(self, num_labels=40):
        super(ResNet50MultiLabel, self).__init__()
        self.base_model = models.resnet50(pretrained=True)  # Load pretrained ResNet50
        num_features = self.base_model.fc.in_features
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

        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for multi-label output

    def forward(self, x):
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

def main(args):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset = CelebADataset(args.root_dir, args.attr_file, transform)
    train_size = int(0.8 * len(dataset))
    train_set, val_set = random_split(dataset, [train_size, len(dataset) - train_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    model = ResNet50MultiLabel().to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCELoss()

    train_validate_loop(model, train_loader, val_loader, args.device, optimizer, criterion,
                        args.epochs, 'resnet50_multilabel_model.pth')

if __name__ == "__main__":
    args = parse_arguments()
    main(args)