"""
Details
"""
import torch
import torch.nn as nn

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
    
def get_awl_optimizer(params_list):
    """ details """
    optimizer = torch.optim.Adam(params_list)
    return optimizer