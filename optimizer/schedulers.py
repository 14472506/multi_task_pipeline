"""
Detials
"""
# imports
import torch

class SchedulerSelector():
    def __init__(self, cfg, optimizer):
        self.cfg = cfg
        self.optimizer = optimizer

    def get_scheduler(self):
        """
        Details
        """
        if self.cfg["sched_name"] == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                        step_size=self.cfg["sched_step"],
                                        gamma=self.cfg["sched_gamma"])
            return scheduler
