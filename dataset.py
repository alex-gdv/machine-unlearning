from torch.utils.data import Dataset
class UTKFace(Dataset):
    def __init__(self, root_dir, device):
        import torch
        from torchvision.models import ResNet50_Weights
        import os
        import random

        self.root_dir = root_dir
        self.device = device

        self.image_paths = os.listdir(self.root_dir)

        self.image_labels = torch.Tensor(
            [int(x.split("_")[0])//10 for x in self.image_paths]
        ).to(torch.int64)

        self.num_classes = int(self.image_labels.max()) + 1

        self.image_ohe = torch.nn.functional.one_hot(
            self.image_labels, num_classes=self.num_classes
        ).to(torch.float32).to(self.device)

        self.image_transform = ResNet50_Weights.DEFAULT.transforms()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        from torchvision.io import read_image

        img = read_image(f"{self.root_dir}/{self.image_paths[idx]}")
        img_tensor = self.image_transform(img).to(self.device)
        label = self.image_ohe[idx]

        return img_tensor, label
