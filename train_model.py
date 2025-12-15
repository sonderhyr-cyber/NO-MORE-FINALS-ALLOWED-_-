import torch
import torch.nn as nn
import torchvision.models as models

class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Sequential(
            nn.Linear(512, 1),
            nn.Dropout(0.4),
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = self.fc(x.view(batch_size, -1))
        return x
