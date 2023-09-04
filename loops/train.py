"""
MainLoop for Training & Testing Models

The script acts as a high-level pipeline manager, encapsulating the whole process of training
the model based on the configuration provided. It loads the configuration, sets up the model, 
optimizer, scheduler, datasets, and logs, and then handles the training, validation, and testing loops.

Last Edited by: Bradley Hurst
"""
# imports
# base packages

# third party packages
import torch

# local packages
from models import Models
from data_handling import Loaders
from losses import Losses
from optimisers import Optimisers, Schedulers
from .logs.logs import Logs
from .actions.preloop import PreLoop
from .actions.prestep import PreStep
from .actions.step import Step
from .actions.poststep import PostStep
from .actions.postloop import PostLoop 

# class
class Train():
    def __init__(self, cfg):
        """Initialize the main loop with given config and seed."""
        self.cfg = cfg
        self.iter_count = 0

        # collect config data
        self._extract_configs()

        # initialise key training components
        self._initialise_model()
        self._initialise_dataloader()
        self._initialise_losses()
        self._initialise_optimiser()
        self._initialise_actions()        
        self._initialise_logs()

    def train(self):
        """ detials """
        self._before_loop()
        
        for epoch in range(self.start, self.end):
            
            self._pre_step(epoch)

            self._step(self.model,
                       self.train_loader,
                       self.val_loader,
                       self.loss, 
                       self.optimiser, 
                       self.device, 
                       self.grad_acc,
                       epoch,
                       self.logs,
                       self.iter_count,
                       self.logger
                       )
            
            self._post_step(epoch, 
                            self.model, 
                            self.optimiser, 
                            self.scheduler, 
                            self.logs, 
                            self.logger)
            
        self._post_loop()

    def _extract_configs(self):
        """ Extract all required parameters from config """
        # loop parameters
        self.device = self.cfg["loops"]["device"]
        self.grad_acc = self.cfg["loops"]["grad_acc"]
        self.start = self.cfg["loops"]["start"]
        self.end = self.cfg["loops"]["end"]

        # sub configs
        self.model_cfg = self.cfg["model"]
        self.loader_cfg = self.cfg["dataset"]
        self.loss_cfg = self.cfg["losses"]
        self.optimiser_cfg = self.cfg["optimiser"]
        self.logs_cfg = self.cfg["logs"]

    def _initialise_model(self):
        """ Detials """
        self.model = Models(self.model_cfg).model()
        self.model.to(self.device)

    def _initialise_dataloader(self):
        """ Detials """
        self.train_loader, self.val_loader = Loaders(self.loader_cfg, "train").loader()

    def _initialise_losses(self):
        """ Detials """
        self.loss = Losses(self.loss_cfg).loss()

    def _initialise_optimiser(self):
        """ Details """
        self.optimiser = Optimisers(self.optimiser_cfg, self.model, self.loss).optimiser()
        if self.optimiser_cfg["sched_name"]:
            self.scheduler = Schedulers(self.optimiser_cfg, self.optimiser).scheduler()
        else:
            self.scheduler = None

    def _initialise_logs(self):
        """ Detials """
        self.logger = Logs(self.logs_cfg)
        self.iter = self.logger.get_iter()
        self.logs = self.logger.get_log()
        self.logger.init_log_file(self.cfg, self.logs)

    def _initialise_actions(self):
        """ Details """
        self._before_loop = PreLoop(self.model_cfg).action()
        self._pre_step = PreStep(self.model_cfg).action()
        self._step = Step(self.model_cfg).action()
        self._post_step = PostStep(self.model_cfg).action()
        self._post_loop = PostLoop(self.model_cfg).action()
        











        







