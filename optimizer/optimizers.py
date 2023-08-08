import torch
import numpy as np
import random
import os

class OptimiserSelector():
    def __init__(self, cfg, model, seed=42):
        self.cfg = cfg
        self.model = model
        self.seed = seed
        self.set_seed()

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

    def _get_adam_optimizer(self):
        """Retrieve the Adam optimizer based on the configuration."""
        return torch.optim.Adam(self.model.parameters(), lr=self.cfg["opt_lr"])

    def _validate_config(self, opt_name):
        """Validate the configuration for the optimizer."""
        # Placeholder for possible validation logic based on specific optimizer
        # e.g. if opt_name == 'SpecificOptimizer': assert 'specific_key' in self.cfg
        pass

    def get_optimizer(self):
        """Retrieve the optimizer based on the configuration."""
        optimizer_mapping = {
            'Adam': self._get_adam_optimizer
            # Add other optimizers here as needed
        }
        
        opt_name = self.cfg["opt_name"]
        
        if opt_name not in optimizer_mapping:
            raise ValueError(f"Optimizer '{opt_name}' not supported. Add the optimizer method to the class and update the mapping.")
        
        self._validate_config(opt_name)
        return optimizer_mapping[opt_name]()

















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

