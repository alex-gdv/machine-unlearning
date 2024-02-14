from torchvision.models import ResNet50_Weights
from torch.utils.data import Dataset
from torchvision.io import read_image
import torch
import os

class UTKFace(Dataset):
    def __init__(self, root_dir, device):
        self.root_dir = root_dir
        self.device = device

        self.image_paths = os.listdir(self.root_dir)

        self.image_labels = torch.Tensor(
            [int(x.split("_")[0]) for x in self.image_paths]
        ).to(torch.float32).to(self.device)

        self.image_transform = ResNet50_Weights.DEFAULT.transforms()


    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, idx):
        img = read_image(f"{self.root_dir}/{self.image_paths[idx]}")
        img_tensor = self.image_transform(img).to(self.device)
        label = self.image_labels[idx].reshape([1])

        return img_tensor, label