"""
Details
"""
# imports
from .classification.rotnet_classifier_model import RotNet

# class
class Models():
    """ Detials """
    def __init__(self, cfg):
        """ Detials """
        self.cfg = cfg
        self._exctract_cfg()
        self._initialise_mapping()

    def _initialise_mapping(self):
        """ Detials """
        self.model_mapping = {
            "rotnet_resnet_50": RotNet
        }
    
    def _exctract_cfg(self):
        """ Detials """
        self.model_name = self.cfg["model_name"]

    def model(self):
        """ Detials """
        if self.model_name in self.model_mapping:
            return self.model_mapping[self.model_name](self.cfg)
        else:
            raise ValueError(f"Model '{self.model_name}' is not recognised. Please check the model name or update the model mapping.")
