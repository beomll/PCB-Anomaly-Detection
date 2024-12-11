import os
import torch
from torch.utils.data import Dataset
from PIL import Image


class AnomalyDataset(Dataset):
    def __init__(self, data_path, transform=None):
        """
        structure of data:
            data_path
                ├── True
                │     ├── 0.jpg
                │     ├── 1.jpg
                │     ├── ...
                ├── False
                │     ├── 0.jpg
                │     ├── 1.jpg
                │     ├── ...

        :param data_path: path to the data directory
        :param transform: transforms to apply to the images
        """

        self.data_path = data_path
        self.transform = transform
        self.labels = {0: 'False', 1: 'True'}

        self.data = []
        for label in self.labels.values():
            label_path = os.path.join(self.data_path, label)
            for img in os.listdir(label_path):
                self.data.append((os.path.join(label_path, img), int(label == 'True')))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, label
