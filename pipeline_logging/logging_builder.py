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