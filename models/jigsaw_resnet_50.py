"""
Detials
"""
# imports
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import torch

# class
class Jigsaw_ResNet_50(nn.Module):
    """
    Details
    """
    def __init__(self, cfg):
        """
        Detials
        """
        super().__init__()
        # TO BE IMPLEMENTED GENERALLY
        #self.backbone = self.backbone_selector(cfg["backbone_type"])
        self.backbone = resnet50(weights=ResNet50_Weights)

        self.num_tiles = cfg["num_tiles"]
        self.num_permutations = cfg["num_permutations"]

        self.twin_network = nn.Sequential(nn.Linear(1000, 512, bias=False),
                                          nn.BatchNorm1d(512),
                                          nn.ReLU(inplace=True))
        
        self.classifier = nn.Sequential(nn.Linear(512*self.num_tiles, 4096, bias=False),
                                         nn.BatchNorm1d(4096),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(4096, self.num_permutations))

    def forward(self, x):
        """
        Detials
        """
        assert x.shape[1] == self.num_tiles
        device = x.device
        x = torch.stack([self.twin_network(self.backbone(tile)) for tile in x]).to(device)
        x = torch.flatten(x, start_dim = 1)
        x = self.classifier(x)

        return x
