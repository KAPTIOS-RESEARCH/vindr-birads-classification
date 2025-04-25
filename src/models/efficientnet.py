from torch import nn
from torchvision import models


class EfficientNetB0(nn.Module):
    def __init__(self, n_output=1):
        super(EfficientNetB0, self).__init__()
        self.model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_features, n_output)

    def forward(self, x):
        return self.model(x)
