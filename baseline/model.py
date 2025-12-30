import torch.nn as nn

from torchvision.models import resnet18


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()

        self.backbone = resnet18(weights=None)

        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        return self.backbone(x)
