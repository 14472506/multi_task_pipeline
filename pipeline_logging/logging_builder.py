class LogBuilder():
    """
    This class provides structured dictionaries for logging based on the configuration provided.
    It helps create and return the right logger structure depending on the logger type requested.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.log = self._log_selector()
    
    def get_logger(self):
        """ Returns the logger structure """
        return self.log

    def _log_selector(self):
        """
        Based on the configuration, selects the appropriate logger structure.
        """
        selector = {
            "base_logger": self._base_logger,
            "multi_task_logger": self._multi_task_logger
        }

        return selector.get(self.cfg["logger"], self._unrecognized_logger)()

    def _base_logger(self):
        """
        Returns a logger structure suitable for basic logging.
        """
        logger = {
            "train_loss": [],
            "val_loss": [],
            "epochs": [],
            "best_val": [],
            "best_epoch": []
        }
        return logger

    def _multi_task_logger(self):
        """
        Returns a logger structure suitable for multi-task logging.
        """
        logger = {
            # epoch count
            "epochs": [],            
            # training losses
            "train_loss": [],
            "ssl_train_loss": [],
            # validation losses and values
            "val_loss": [],
            "ssl_val_loss": [],
            "mAP": [],
            # best results
            "best_mAP": [],
            "best_mAP_epoch": [],
            "best_mAP_step_epoch": [],
            "best_val": [],
            "best_epoch": [],
            "best_step_epoch": [],
        }
        return logger

    def _unrecognized_logger(self):
        """
        Returns a basic logger structure and prints a warning for unrecognized logger type.
        """
        print(f"Warning: Unrecognized logger type '{self.cfg['logger']}'. Defaulting to 'base_logger'.")
        return self._base_logger()




















"""
Details
"""
# imports

# class
class LogBuilder():
    """
    Detials
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.log = self.log_selector()
    
    def get_logger(self):
        return(self.log)

    def log_selector(self):
        """
        Details
        """
        if self.cfg["logger"] == "base_logger":
            logger = self.base_logger()
            return logger
        if self.cfg["logger"] == "multi_task_logger":
            logger = self.mutli_task_logger()
            return logger

    def base_logger(self):
        """
        Details
        """
        logger = {
            "train_loss": [],
            "val_loss": [],
            "epochs": [],
            "best_val": [],
            "best_epoch": []
        }
        return logger

    def mutli_task_logger(self):
        """
        Details
        """
        logger = {
            # epoch count
            "epochs": [],            
            # training losses
            "train_loss": [],
            "ssl_train_loss": [],
            # validation losses and values
            "val_loss": [],
            "ssl_val_loss": [],
            "mAP": [],
            # best results
            "best_mAP": [],
            "best_mAP_epoch": [],
            "best_mAP_step_epoch": [],
            "best_val": [],
            "best_epoch": [],
            "best_step_epoch": [],
        }
        return logger