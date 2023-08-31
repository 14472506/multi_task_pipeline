import torch
import numpy as np
import random
import os

class Optimisers():
    """ Detials """
    def __init__(self, cfg, model):
        """ Detials """
        self.cfg = cfg
        self.model = model
        self._extract_config()
        self._get_model_params()


    def _extract_config(self):
        """ Detials """
        self.model_name = self.cfg["model_name"]
        self.opt_name = self.cfg["opt_name"]
        self.params = self.cfg["opt_params"]

    def optimiser(self):
        """ Detials """
        optimiser_map = {
            "Adam": self._get_adam_optimizer
        }
        return optimiser_map[self.opt_name]()

    def _get_adam_optimizer(self):
        """Retrieve the Adam optimizer based on the configuration."""
        return torch.optim.Adam(self.model_params)
    
    def _get_model_params(self):
        """ Detials """
        if self.model_name == "rotnet_resnet_50":
            self.model_params = [{"params": self.model.parameters(), "lr": self.params["lr"]}]
        



        
