import torch
import torch.nn as nn
from torchvision import models

def load_classifier(weights_path, model_name, device):
    classifier = MultiLabelModel(model_name=model_name, num_labels=40).to(device)
    classifier.load_state_dict(torch.load(weights_path, map_location=device))
    classifier.eval()
    return classifier

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
            raise ValueError(f"Model '{model_name}' is not supported.")

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