"""
Detials
"""
# import
import torch
import torch.nn.functional as F

# class
class Losses():
    """ Detials """
    def __init__(self, cfg):
        """ Detials """
        self.cfg = cfg
        self.model_type = self.cfg["model_name"]
    
    def loss(self):
        loss_mapper = {
            "rotnet_resnet_50": self._classifier_loss
        }
        return loss_mapper[self.model_type]
    
    def _classifier_loss(self, target, pred):
        """ Detials """
        loss = F.cross_entropy(target, pred)
        return loss
