"""
Detials
"""
# imports
import torch
import numpy as np
import random
import os

class OptimiserSelector():
    def __init__(self, cfg, model, seed=42):
        self.cfg = cfg
        self.model = model
        self.seed=seed

    def set_seed(self):
        """
        Details
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(self.seed)

    def get_optimizer(self):
        """
        Detials
        """
        if self.cfg["opt_name"] == "Adam":
            optimizer =  torch.optim.Adam(self.model.parameters(), lr=self.cfg["opt_lr"])
            return optimizer

