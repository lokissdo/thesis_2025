import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from utils.args import parse_arguments
from utils.dataset import CelebADataset
from utils.train_utils import train_validate_loop

class DenseNet121MultiLabel(nn.Module):
    def __init__(self, num_labels=40):
        super().__init__()
        self.base_model = models.densenet121(pretrained=True)
        num_features = self.base_model.classifier.in_features
        self.base_model.classifier = nn.Linear(num_features, num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.base_model(x))

def main(args):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset = CelebADataset(args.root_dir, args.attr_file, transform)
    train_size = int(0.8 * len(dataset))
    train_set, val_set = random_split(dataset, [train_size, len(dataset) - train_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    model = DenseNet121MultiLabel().to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCELoss()

    train_validate_loop(model, train_loader, val_loader, args.device, optimizer, criterion,
                        args.epochs, 'densenet121_multilabel_model.pth')

if __name__ == "__main__":
    args = parse_arguments()
    main(args)