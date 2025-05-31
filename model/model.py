from torch import nn
from torchvision import models


class ResNetRegression(nn.Module):
    def __init__(self):
        super(ResNetRegression, self).__init__()

        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.nr_features = self.model.fc.in_features
        self.model.fc = nn.Linear(self.nr_features, 1)
        
    def reset_first_k_layers(self, k=0):
        for i, layer in enumerate(self.model.children()):
            if isinstance(layer, nn.Conv2d) and i < k:
                layer.reset_parameters()
            
    def forward(self, x):
        x = self.model(x)

        return x.squeeze()
    

class ResNet50Ordinal(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50Ordinal, self).__init__()

        self.num_classes = num_classes 

        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.nr_features = self.model.fc.in_features
        self.model.fc = nn.Linear(self.nr_features, self.num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)

        return self.sigmoid(x).squeeze()
