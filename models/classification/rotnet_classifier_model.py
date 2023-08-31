"""
Detials
"""
# imports
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import torch

# class
class RotNet(nn.Module):
    """ Detials """
    def __init__(self, cfg):
        """ Detials """
        super().__init__()

        # extract configs
        self.cfg = cfg
        self._extract_config()

        # define model
        self._initialise_backbone()
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
        self.classifier = nn.Sequential(*[child for child in self.classifier.children() if not isinstance(child, nn.Identity)])

    def forward(self, x):
        """ Detials """
        x = self.backbone(x)
        x = self.classifier(x)
        return x
    
    def _extract_config(self):
        """ Detials """
        self.model_name = self.cfg["model_name"]
        self.backbone_type = self.cfg["params"]["backbone_type"]
        self.num_rots = self.cfg["params"]["num_rotations"]
        self.drop_out = self.cfg["params"]["drop_out"]
        self.batch_norm = self.cfg["params"]["batch_norm"]
        self.trainable_layers = self.cfg["params"]["trainable_layers"]

    def _initialise_backbone(self):
        """ Detials """
        if self.backbone_type == "pre_trained":
            self.backbone = resnet50(weights=ResNet50_Weights)
        else:
            self.backbone = resnet50()