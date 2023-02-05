"""
Detials
"""
# imports
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import torch

# class
class RotNet(nn.Module):
    """
    Detials
    """
    def __init__(self, cfg, num_rotations=4, dropout_rate=0.5, batch_norm=True):
        """
        Detials
        """
        self.backbone = self.backbone_selector(cfg["model"]["backbone"])

        self.classifier = nn.Sequential(nn.Dropout() if dropout_rate > 0. else nn.Identity(),
                                nn.Linear(1000, 4096, bias=False if batch_norm else True),
                                nn.BatchNorm1d(4096) if batch_norm else nn.Identity(),
                                nn.ReLU(inplace=True),
                                nn.Dropout() if dropout_rate > 0. else nn.Identity(),
                                nn.Linear(4096, 4096, bias=False if batch_norm else True),
                                nn.BatchNorm1d(4096) if batch_norm else nn.Identity(),
                                nn.ReLU(inplace=True),
                                nn.Dropout() if dropout_rate > 0. else nn.Identity(),
                                nn.Linear(4096, 1000, bias=False if batch_norm else True),
                                nn.BatchNorm1d(1000) if batch_norm else nn.Identity(),
                                nn.ReLU(inplace=True),
                                nn.Linear(1000, num_rotations))

                # Remove any potential nn.Identity() layers
        self.classifier = nn.Sequential(*[child for child in self.classifier.children() if not isinstance(child, nn.Identity)])

    def forward(self, x):
        """
        Detials
        """
        x = self.backbone(x)
        x = self.classifier(x)
        return x

    def backbone_selector(self, pre_trained):
        """
        Detials
        """
        # either load pre-trained weights or dont
        if pre_trained == "pre_trained":
            backbone = resnet50(weights=ResNet50_Weights)
        elif pre_trained == "load":
            backbone = backbone_loader(self.cd["model"]["load_model"])
        else:
            backbone = resnet50()
        
        # return backbone
        return backbone