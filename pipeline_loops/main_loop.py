"""
Details
"""
# imports
import torch
from models import ModelBuilder
from optimizer import OptimiserSelector, SchedulerSelector
from datasets import DataloaderBuilder
from .loop_selector import LoopSelector
from pipeline_logging import LogBuilder
from utils import (best_loss_saver, save_model, save_json,
                schedule_loader, load_model, garbage_collector)
from utils.coco_evaluation import evaluate

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
        self.scheduler = None
        if "sched_name" in cfg["optimizer"]:
            self.scheduler = SchedulerSelector(cfg["optimizer"], self.optimizer).get_scheduler()
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg["loop"]["amp"])

    
        # init logger
        self.logger = LogBuilder(cfg["logging"]).get_logger()        
        
        # data loader and loops
        if cfg["loop"]["train"]:
            # data loaders
            self.train_loader = DataloaderBuilder(cfg, "train").loader()
            self.val_loader = DataloaderBuilder(cfg, "val").loader()
            # loops
            self.train_loop = LoopSelector(cfg["loop"]).get_train()
            self.val_loop = LoopSelector(cfg["loop"]).get_val()
              
        if cfg["loop"]["test"]:
            # data laoder
            self.test_loader = DataloaderBuilder(cfg, "test").loader()
            # loop 
            self.test_loop = LoopSelector(cfg["loop"]).get_test()
        
    def train(self):

        save_json(self.cfg,
                self.cfg["logging"]["path"],
                "cfg.json")

        best_loss = 100 # arbitraraly high
        best_mAP = 0 # arb low
        for epoch in range(self.cfg["loop"]["start_epoch"], self.cfg["loop"]["end_epoch"]):
            
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
                self.scaler,
                self.logger,
                self.cfg["loop"]["device"],
                self.iter_count,
                epoch)

            garbage_collector()
            
            print(" --- Validation ----------------------------------------------------------------")

            # validation loop
            self.val_loop(self.model,
                self.val_loader,                
                self.scaler,
                self.logger,
                self.cfg["loop"]["device"],
                epoch,
                self.cfg["logging"]["path"])

            garbage_collector()

            # model saving
            save_model(epoch, 
                self.model, 
                self.optimizer, 
                self.cfg["logging"]["path"], 
                "last_model.pth")

            best_loss = best_loss_saver(epoch,
                self.logger, 
                self.model,
                self.optimizer,
                self.cfg["logging"]["path"],
                self.cfg["logging"]["pth_name"],
                best_loss)

            # TIDY THIS UP
            if self.logger["mAP"][-1] > best_mAP:
                save_model(epoch, 
                    self.model, 
                    self.optimizer, 
                    self.cfg["logging"]["path"], 
                    "best_mAP.pth")
                best_mAP = self.logger["mAP"][-1]
                self.logger["best_mAP"].append(best_mAP)
                self.logger["best_mAP_epoch"].append(epoch)

            # log saving
            save_json(self.logger,
                self.cfg["logging"]["path"],
                "log.json")   

            if "sched_name" in self.cfg["optimizer"]:
                if epoch == self.cfg["optimizer"]["sched_step"] -1:
                    schedule_loader(self.model,
                        self.cfg["logging"]["path"],
                        self.cfg["logging"]["pth_name"])
                    best_loss = 100 # arbitrarally high
                # this is going last
                self.scheduler.step()

            garbage_collector()

    def test(self):

        # load model
        load_model(self.cfg["logging"]["path"],
            self.cfg["logging"]["pth_name"],
            self.model)

        print("================================================================================")
        print(" Evaluating")
        print("================================================================================")
        print(" --- Best Model ----------------------------------------------------------------")

        self.test_loop(self.model, 
            self.test_loader, 
            self.cfg["loop"]["device"], 
            self.cfg["logging"]["path"], 
            train_flag=True)

        garbage_collector()
        
        # if step scheduler is used
        if "sched_name" in self.cfg["optimizer"]:
            load_model(self.cfg["logging"]["path"],
                "/ps_" + self.cfg["logging"]["pth_name"],
                self.model)

            print(" --- Pre-Step Best Model -------------------------------------------------------")
            self.test_loop(self.model,
                self.test_loader, 
                self.cfg["loop"]["device"], 
                self.cfg["logging"]["path"], 
                train_flag=True)
            
            

        
        


                        
            

            










    

    