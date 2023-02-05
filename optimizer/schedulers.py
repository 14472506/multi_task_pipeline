"""
Detials
"""
# imports
import torch
import numpy as np
import random
import os

class SchedulerSelector():
    def __init__(self, cfg, optimizer, seed=42):
        self.cfg = cfg
        self.optimizer = optimizer
        self.seed = seed

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

    def get_scheduler(self):
        """
        Details
        """
        if self.cfg["sched_name"] == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                        step_size=self.cfg["sched_step"],
                                        gamma=self.cfg["sched_gamma"])
            return scheduler
    

