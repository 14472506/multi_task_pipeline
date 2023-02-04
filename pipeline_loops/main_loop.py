"""
Details
"""
# imports
from models import ModelBuilder
from optimizer import OptimiserSelector, SchedulerSelector
from datasets import DataloaderBuilder
from .loop_selector import LoopSelector
from pipeline_logging import LogBuilder
from utils import best_loss_saver, save_model, save_json

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
        self.iter_count = 0
        
        # model
        self.model = ModelBuilder(cfg["model"]).model_builder()
        self.model.to(cfg["loop"]["device"])
        
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
        
    def train(self):

        save_json(self.cfg,
                self.cfg["logging"]["path"],
                "cfg.json")

        for epoch in range(10):
            
            # record epoch
            self.logger["epochs"].append(epoch)
            
            # training loop
            print("================================================================================")
            print(" [Epoch: %s]" %epoch)
            print("================================================================================")
            print(" --- Training ------------------------------------------------------------------")
            
            self.iter_count = self.train_loop(self.model, 
                                    self.train_loader,
                                    self.optimizer,
                                    self.logger,
                                    self.cfg["loop"]["device"],
                                    self.iter_count,
                                    epoch)
            
            print(" --- Validation ----------------------------------------------------------------")

            # validation loop
            self.val_loop(self.model,
                    self.val_loader,
                    self.logger,
                    self.cfg["loop"]["device"],
                    epoch)

            # model saving
            save_model(epoch, 
                    self.model, 
                    self.optimizer, 
                    self.cfg["logging"]["path"], 
                    "last_model.pth")

            best_loss_saver(epoch,
                    self.logger, 
                    self.model,
                    self.optimizer,
                    self.cfg["logging"]["path"],
                    self.cfg["logging"]["pth_name"])

            save_json(self.logger,
                    self.cfg["logging"]["path"],
                    "log.json")

                        
            

            










    

    