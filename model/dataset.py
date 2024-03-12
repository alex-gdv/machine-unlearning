from torchvision.models import ResNet50_Weights
from torch.utils.data import Dataset
from torchvision.io import read_image
import numpy as np
import json


class UTKFaceRegression(Dataset):
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
    

class UTKFaceOrdinal(Dataset):
    def __init__(self, meta_path, num_classes):
        self.num_classes=num_classes

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

        ordinal_encoding = np.zeros(self.num_classes)
        ordinal_encoding[:label//self.num_classes] = 1

        return img_tensor, ordinal_encoding
