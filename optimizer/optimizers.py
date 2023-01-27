"""
Detials
"""
# imports
import torch

class OptimiserSelector():
    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model = model

    def get_optimizer(self):
        """
        Detials
        """
        if self.cfg["opt_name"] == "Adam":
            optimizer =  torch.optim.Adam(self.model.parameters(), lr=self.cfg["opt_lr"])
            return optimizer

