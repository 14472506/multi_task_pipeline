import torch
import numpy as np
import random
import os

class Schedulers():
    """ Detials """
    def __init__(self, cfg, optimizer):
        self.optimizer = optimizer
        self.cfg = cfg
        self._extract_cfg()

    def _extract_cfg(self):
        """ Detials """
        self.sched_name = self.cfg["sched_name"]
        self.sched_step = self.cfg["sched_params"]["step"]
        self.sched_gamma = self.cfg["sched_params"]["gamma"]

    def _get_step_lr_scheduler(self):
        """Retrieve the StepLR scheduler based on the configuration."""
        return torch.optim.lr_scheduler.StepLR(self.optimizer,
                                              step_size=self.sched_step,
                                              gamma=self.sched_gamma)

    def scheduler(self):
        """Retrieve the scheduler based on the configuration."""
        scheduler_mapping = {
            'StepLR': self._get_step_lr_scheduler()
        }
        return scheduler_mapping[self.sched_name]

