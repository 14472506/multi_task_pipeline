"""
Detials
"""
# imports
from models import Models
from data_handling import Loaders
from losses import Losses
from .logs.logs import Logs
from .actions.test import TestAction


# classes
class Test():
    """ Detials """
    def __init__(self, cfg):
        """ Detials """
        self.cfg = cfg
        
        # extract configs
        self._extract_cfg()

        # initialise key test components
        self._initialise_model()
        self._initialise_dataloader()
        self._initialise_optimiser()
        self._initialise_actions()
        self._initialise_logs()

    def test(self):
        """ Detials """
        self.test_action(self.model, self.test_loader, self.step, self.logger, self.device)
    
    def _extract_cfg(self):
        """ Detials """
        self.device = self.cfg["loops"]["device"]
        self.model_cfg = self.cfg["model"]
        self.optimiser_cfg = self.cfg["optimiser"]
        self.logs_cfg = self.cfg["logs"]
        self.loader_cfg = self.cfg["dataset"]

    def _initialise_model(self):
        """ Detials """
        self.model = Models(self.model_cfg).model()
        self.model.to(self.device)

    def _initialise_dataloader(self):
        """ Detials """
        self.test_loader = Loaders(self.loader_cfg, "test").loader()

    def _initialise_optimiser(self):
        """ Detials """
        self.step = True if self.optimiser_cfg["sched_name"] else False

    def _initialise_logs(self):
        """ Detials """
        self.logger = Logs(self.logs_cfg)
        self.iter = self.logger.get_iter()
        self.logs = self.logger.get_log()
        self.logger.init_log_file(self.cfg, self.logs)
    
    def _initialise_actions(self):
        """ Detials """
        self.test_action = TestAction(self.model_cfg).action()