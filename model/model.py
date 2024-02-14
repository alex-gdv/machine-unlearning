from torch import nn

class ResNet50(nn.Module):
    def __init__(self):
        from torchvision.models import resnet50, ResNet50_Weights

        super(ResNet50, self).__init__()

        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.nr_features = self.model.fc.in_features
        self.model.fc = nn.Linear(self.nr_features, 1)

    def forward(self, x):
        x = self.model(x)
        return x
