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
        self.twin_network = nn.Sequential(nn.Linear(1000, 2084, bias=False),
                                          nn.BatchNorm1d(2048),
                                          nn.ReLU(inplace=True),
                                          nn.Dropout(p=0.5),)
        
        self.classifier = nn.Sequential(nn.Linear(1000*self.num_tiles, 4096, bias=False),
                                         nn.BatchNorm1d(4096),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(p=0.5),
                                         nn.Linear(4096, 2048, bias=False),
                                         nn.BatchNorm1d(2048),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(p=0.5),
                                         nn.Linear(2048, 1024, bias=False),
                                         nn.BatchNorm1d(1024),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(p=0.5),
                                         nn.Linear(1024, self.num_permutations))
        
    def _extract_cfg(self):
        """ Detial """
        self.num_tiles = self.cfg["params"]["num_tiles"]
        self.num_permutations = self.cfg["params"]["num_permutations"]
        self.batch_norm = self.cfg["params"]["batch_norm"]

    def forward(self, x):
        """ Detials """
        device = x.device
        assert x.shape[1] == self.num_tiles

        x = torch.stack([self.backbone(tile) for tile in x]).to(device)
        x = torch.flatten(x, start_dim = 1)
        x = self.classifier(x)

        return x
