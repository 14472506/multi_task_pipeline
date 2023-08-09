import torch
import numpy as np
import random
import os

class SchedulerSelector():
    def __init__(self, cfg, optimizer, seed=42):
        self.optimizer = optimizer
        self.seed = seed
        self.set_seed()

        # params
        self.sched_name = cfg["scheduler"]["name"]
        self.sched_step = cfg["scheduler"]["params"]["step"]
        self.sched_gamma = cfg["scheduler"]["params"]["gamma"]

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

    def _get_step_lr_scheduler(self):
        """Retrieve the StepLR scheduler based on the configuration."""
        return torch.optim.lr_scheduler.StepLR(self.optimizer,
                                              step_size=self.sched_step,
                                              gamma=self.sched_gamma)

    #def _validate_config(self):
    #    """Validate the configuration for the scheduler."""
    #    # Placeholder for possible validation logic based on specific scheduler
    #    # e.g. if sched_name == 'SpecificScheduler': assert 'specific_key' in self.cfg
    #    pass

    def get_scheduler(self):
        """Retrieve the scheduler based on the configuration."""
        scheduler_mapping = {
            'StepLR': self._get_step_lr_scheduler
            # Add other schedulers here as needed
        }
        
        if self.sched_name not in scheduler_mapping:
            raise ValueError(f"Scheduler '{self.sched_name}' not supported. Add the scheduler method to the class and update the mapping.")
        
        #self._validate_config(self.sched_name)
        return scheduler_mapping[self.sched_name]()















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
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(self.seed)

    def get_scheduler(self):
        if self.cfg["sched_name"] == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                        step_size=self.cfg["sched_step"],
                                        gamma=self.cfg["sched_gamma"])
            return scheduler
"""