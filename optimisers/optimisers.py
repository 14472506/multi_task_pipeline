""" 
Module Detials: 
The module handles the optimiser selection used in the training
loop
"""
# imports
# base packages
import random
import os

# third party packages
import torch
import numpy as np

# local packages

# class
class Optimisers():
    def __init__(self, cfg, model, loss):
        """
        Initializes the optimizer handler with configuration, 
        model, and loss function. 
        """
        self.cfg = cfg
        self.model = model
        self.loss = loss
        self._extract_config()
        self._get_model_params()

    def _extract_config(self):
        """ Extracts the provided configs """
        self.model_name = self.cfg["model_name"]
        self.opt_name = self.cfg["opt_name"]
        self.params = self.cfg["opt_params"]

    def optimiser(self):
        """ returns the selected optimiser when call """
        optimiser_map = {
            "Adam": self._get_adam_optimizer
        }
        return optimiser_map[self.opt_name]()

    def _get_adam_optimizer(self):
        """ Retrieve the Adam optimizer based on the configuration. """
        return torch.optim.Adam(self.model_params)
    
    def _get_model_params(self):
        """ returns the optimiser paramaters based on the selected model """
        if self.model_name == "rotnet_resnet_50":
            self.model_params = [{"params": self.model.parameters(), "lr": self.params["lr"]}]
        if self.model_name == "rotmask_multi_task":
            self.model_params = [{"params": self.model.parameters(), "lr": self.params["lr"]}, {"params": self.loss[0].parameters()}]
        



        
