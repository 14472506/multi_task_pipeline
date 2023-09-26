"""
Detials
"""
# imports
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import torch

# class
class Jigsaw(nn.Module):
    """ Details """
    def __init__(self, cfg):
        """ Detials """
        super().__init__()

        self.cfg = cfg
        self._extract_cfg()

        # TO BE IMPLEMENTED GENERALLY
        self.backbone = resnet50(weights=ResNet50_Weights)
        self.twin_network = nn.Sequential(nn.Linear(1000, 512, bias=False),
                                          nn.BatchNorm1d(512),
                                          nn.ReLU(inplace=True))
        
        self.classifier = nn.Sequential(nn.Linear(512*self.num_tiles, 4096, bias=False),
                                         nn.BatchNorm1d(4096),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(4096, self.num_permutations))
        
    def extract_cfg(self):
        """ Detial """
        self.num_tiles = self.cfg["params"]["num_tiles"]
        self.num_permutations = self.cfg["params"]["num_permutations"]
        self.batch_norm = self.cfg["params"]["batch_norm"]

    def forward(self, x):
        """ Detials """
        assert x.shape[1] == self.num_tiles
        device = x.device
        x = torch.stack([self.twin_network(self.backbone(tile)) for tile in x]).to(device)
        x = torch.flatten(x, start_dim = 1)
        x = self.classifier(x)

        return x
