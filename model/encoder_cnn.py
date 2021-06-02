import torch
import torch.nn as nn
import torchvision.models as models

class EncoderResNet(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        for p in self.model.parameters(): # freeze all parameters
            p.requires_grad = False;
        # fine-tuning
        self.model.fc = nn.Sequential(
                            nn.Linear(2048, 2*feature_dim),
                            nn.ReLU(inplace=True),
                            nn.Linear(2*feature_dim, feature_dim))
    def forward(self, data):
        return self.model(data)
