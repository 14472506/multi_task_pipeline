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
        if self.cfg["logger"] == "base_instance_logger":
            logger = self.base_instance_logger()
            return logger

    def base_instance_logger(self):
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