"""
Detials
"""
# imports
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import torch

# class
class RotNet_Resnet_50(nn.Module):
    """
    Detials
    """
    def __init__(self, cfg):
        """
        Detials
        """

        super().__init__()
        self.backbone = self.backbone_selector(cfg["params"]["backbone_type"])
        self.drop_out = cfg["params"]["drop_out"]
        self.batch_norm = cfg["params"]["batch_norm"]
        self.num_rots = cfg["params"]["num_rotations"]

        self.classifier = nn.Sequential(nn.Dropout() if self.drop_out > 0. else nn.Identity(),
                                nn.Linear(1000, 4096, bias=False if self.batch_norm else True),
                                nn.BatchNorm1d(4096) if self.batch_norm else nn.Identity(),
                                nn.ReLU(inplace=True),
                                nn.Dropout() if self.drop_out > 0. else nn.Identity(),
                                nn.Linear(4096, 4096, bias=False if self.batch_norm else True),
                                nn.BatchNorm1d(4096) if self.batch_norm else nn.Identity(),
                                nn.ReLU(inplace=True),
                                nn.Dropout() if self.drop_out > 0. else nn.Identity(),
                                nn.Linear(4096, 1000, bias=False if self.batch_norm else True),
                                nn.BatchNorm1d(1000) if self.batch_norm else nn.Identity(),
                                nn.ReLU(inplace=True),
                                nn.Linear(1000, self.num_rots))

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
        else:
            backbone = resnet50()
        
        # return backbone
        return backbone