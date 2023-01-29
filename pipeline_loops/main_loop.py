"""
Details
"""
# imports
from models import ModelBuilder
from optimizer import OptimiserSelector, SchedulerSelector
from datasets import DataloaderBuilder
from .loop_selector import LoopSelector
from pipeline_logging import LogBuilder

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
    
        # init logger
        self.logger = LogBuilder(cfg["logging"]).get_logger()        
        
        # data loader and loops
        if cfg["loop"]["train"]:
            # data loaders
            self.train_loader = DataloaderBuilder(cfg["dataset"]["train"]).coco_loader()
            self.val_loader = DataloaderBuilder(cfg["dataset"]["val"]).coco_loader()
            # loops
            self.train_loop = LoopSelector(cfg["loop"]).get_train()
            self.val_loop = LoopSelector(cfg["loop"]).get_val()
              
        if cfg["loop"]["test"]:
            # data laoder
            self.test_loader = DataloaderBuilder(cfg["dataset"]["test"]).coco_loader()
            # loop 
            self.test_loop = LoopSelector(cfg["loop"]).get_test()
        
        self.train_loop(3,2)










    

    