import torch
import torch.nn as nn
from torchvision import models

def load_classifier(weights_path, model_name, device):
    model = MultiLabelModel(model_name=model_name, num_labels=40).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

class MultiLabelModel(nn.Module):
    def __init__(self, model_name='resnet50', num_labels=40):
        super(MultiLabelModel, self).__init__()
        if model_name == 'resnet50':
            base = models.resnet50(pretrained=True)
            num_features = base.fc.in_features
            base.fc = nn.Identity()
            self.base_model = base
            self.fc = nn.Sequential(
                nn.Linear(num_features, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
                nn.Linear(512, num_labels)
            )
        else:
            raise ValueError(f"Model '{model_name}' is not supported.")
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.base_model(x)
        x = self.fc(x)
        return self.sigmoid(x)