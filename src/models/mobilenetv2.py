from torch import nn
from torchvision import models

class MobileNetV2Custom(nn.Module):
    def __init__(self, n_classes: int = 2):
        super(MobileNetV2Custom, self).__init__()
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.model.classifier[1] = nn.Linear(in_features=1280, out_features=n_classes)
    
    def forward(self, x):
        return self.model(x)