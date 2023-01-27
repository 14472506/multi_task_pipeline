"""
Details
"""
# imports
from models import ModelBuilder
from optimizer import OptimiserSelector, SchedulerSelector
from datasets import DataloaderBuilder

# classes
class MainLoop():
    """
    Details
    """
    def __init__(self, cfg, seed=42):
        """
        Details
        """
        # config
        self.cfg = cfg
        
        # model
        self.model = ModelBuilder(cfg["model"]).model_builder()
        
        # optimizer and scheduler
        self.optimizer = OptimiserSelector(cfg["optimizer"], self.model).get_optimizer()
        if "sched_name" in cfg["optimizer"]:
            self.scheduler = SchedulerSelector(cfg["optimizer"], self.optimizer).get_scheduler()
        
        # data loader
        if cfg["loop"]["train"]:
            self.train_loader = DataloaderBuilder(cfg["dataset"]["train"]).coco_loader()
            self.val_loader = DataloaderBuilder(cfg["dataset"]["val"]).coco_loader()
        if cfg["loop"]["test"]:
            self.test_loader = DataloaderBuilder(cfg["dataset"]["test"]).coco_loader()

        # hooks?
        self.train_loop = 
        self.val_loop = 
        self.loop_logging =
        self.loop_saver =

    

    