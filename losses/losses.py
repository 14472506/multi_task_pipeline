"""
Detials
"""
# import
import torch
import torch.nn.functional as F
import torch.nn as nn

# class
class Losses():
    """ Detials """
    def __init__(self, cfg):
        """ Detials """
        self.cfg = cfg
        self.model_type = self.cfg["model_name"]
    
    def loss(self):
        if self.model_type == "rotnet_resnet_50": 
            return self._classifier_loss,
        if self.model_type == "rotmask_multi_task": 
            return [self._AWL(), self._classifier_loss]

    def _classifier_loss(self, target, pred):
        """ Detials """
        loss = F.cross_entropy(target, pred)
        return loss
    
    def _AWL(self):
        """ Detials """
        loss = AutomaticWeightedLoss()
        return loss

class AutomaticWeightedLoss(nn.Module):
    """ Detials """
    def __init__(self, num_losses=3):
        super().__init__()
        params = torch.ones(num_losses, requires_grad=True)
        self.params = torch.nn.Parameter(params)
    
    def forward(self, *x):
        """ Detials """
        losses_sum = 0
        for i, loss in enumerate(x):
            losses_sum += 0.5/(self.params[i]**2) * loss + torch.log(1+self.params[i]**2)
        return losses_sum