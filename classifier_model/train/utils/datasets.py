import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class CelebADataset(Dataset):
    def __init__(self, root_dir, attr_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.attrs = pd.read_csv(attr_file, delim_whitespace=True, skiprows=1).iloc[:, :40]
        self.attrs.replace(-1, 0, inplace=True)
        self.images = self.attrs.index.tolist()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        labels = self.attrs.iloc[idx].values.astype(np.float32)
        return image, labels