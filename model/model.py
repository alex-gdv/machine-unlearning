from torch import nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50Regression(nn.Module):
    def __init__(self):
        super(ResNet50Regression, self).__init__()

        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.nr_features = self.model.fc.in_features
        self.model.fc = nn.Linear(self.nr_features, 1)

    def forward(self, x):
        x = self.model(x)

        return x.squeeze()
    

class ResNet50Ordinal(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50Ordinal, self).__init__()

        self.num_classes = num_classes 

        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.nr_features = self.model.fc.in_features
        self.model.fc = nn.Linear(self.nr_features, self.num_classes)
        self.sigmomid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)

        return self.sigmoid(x).squeeze()
