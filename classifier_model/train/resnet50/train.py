import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from utils.args import parse_arguments
from utils.dataset import CelebADataset
from utils.train_utils import train_validate_loop

class ResNet50MultiLabel(nn.Module):
    def __init__(self, num_labels=40):
        super().__init__()
        base = models.resnet50(pretrained=True)
        num_features = base.fc.in_features
        base.fc = nn.Identity()

        self.backbone = base
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_labels),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)

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