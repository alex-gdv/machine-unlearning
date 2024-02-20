from torchvision.models import ResNet50_Weights
from torch.utils.data import Dataset
from torchvision.io import read_image
import torch
import json
import os


class UTKFace(Dataset):
    def __init__(self, meta_path):
        with open(meta_path, "r") as fp:
            self.meta = json.load(fp)
        
        self.filenames = list(self.meta.keys())
        self.image_transform = ResNet50_Weights.DEFAULT.transforms()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img = read_image(filename)

        img_tensor = self.image_transform(img)
        label = self.meta[filename]["label"]

        return img_tensor, label
