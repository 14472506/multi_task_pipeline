"""
MainLoop for Training & Testing Models

The script acts as a high-level pipeline manager, encapsulating the whole process of training
the model based on the configuration provided. It loads the configuration, sets up the model, 
optimizer, scheduler, datasets, and logs, and then handles the training, validation, and testing loops.

Last Edited by: Bradley Hurst
"""

# Standard Libraries
import logging

# Third-party Libraries
import torch

# Local Imports
from models import ModelBuilder
from optimizer import OptimiserSelector, SchedulerSelector
from optimizer import AutomaticWeightedLoss, get_awl_optimizer
from datasets import DataloaderBuilder
from .loop_selector import LoopSelector
from pipeline_logging import LogBuilder
from utils import (best_loss_saver, save_model, save_json, 
                   schedule_loader, load_model, garbage_collector)
from utils.coco_evaluation import evaluate

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(filename="pipeline.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MainLoop:
    def __init__(self, cfg, seed=42):
        """Initialize the main loop with given config and seed."""
        self.cfg = cfg
        self.iter_count = 0
        self.best_loss = float('inf')
        
        # Extract configurations
        self._extract_configurations()

        # Initialize main components
        self._initialize_components()

    def _extract_configurations(self):
        """Extract necessary configurations."""
        self.model_cfg = self.cfg.get("model", {})
        self.logging_cfg = self.cfg.get("logging", {})
        self.optimizer_cfg = self.cfg.get("optimizer", {})
        self.device = self.cfg.get("loop", {}).get("device", "cpu")
        self.amp = self.cfg.get("loop", {}).get("amp", False)
        self.actions = self.cfg.get("loop", {}).get("actions", [])
        self.path = self.logging_cfg.get("path", "./")
        self.start = self.cfg.get("loop", {}).get("epochs", {}).get("start", 0)
        self.end = self.cfg.get("loop", {}).get("epochs", {}).get("end", 0)
        self.path_name = self.logging_cfg.get("checkpoint_name", "checkpoint")

    def _initialize_components(self):
        """Initialize the components (model, optimizer, datasets) based on the config."""
        self._initialize_model()
        # if multi_task init AWL
        if self.cfg["loop"]["type"] == "multi_task":
            self._initialize_awl()
        self._initialize_optimizer_and_scheduler()
        self.logger = LogBuilder(self.logging_cfg).get_logger()
        self._initialize_data_loops()

    def _initialize_model(self):
        """Initialize the model."""
        self.model = ModelBuilder(self.model_cfg).model_builder()
        self.model.to(self.device)

    def _initialize_optimizer_and_scheduler(self):
        """Initialize optimizer and scheduler based on config."""
        if self.cfg["loop"]["type"] == "multi_task":
            params_list = [{"params": self.model.parameters(), "lr": self.optimizer_cfg["params"]["lr"]}, {"params": self.awl.parameters()}]
            self.optimizer = get_awl_optimizer(params_list)
        else:
            self.optimizer = OptimiserSelector(self.optimizer_cfg, self.model).get_optimizer()

        if "scheduler" in self.optimizer_cfg:
            self.scheduler = SchedulerSelector(self.optimizer_cfg, self.optimizer).get_scheduler()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

    def _initialize_data_loops(self):
        """Initialize data loaders and train/val/test loops based on the config."""
        # TODO: Address the unused cfg["loop"]["loop_type"]
        if "train" in self.actions:
            self.train_loader = DataloaderBuilder(self.cfg, "train").loader()
            self.val_loader = DataloaderBuilder(self.cfg, "val").loader()
            self.train_loop = LoopSelector(self.cfg["loop"]).get_train()
            self.val_loop = LoopSelector(self.cfg["loop"]).get_val()
        if "test" in self.actions:
            self.test_loader = DataloaderBuilder(self.cfg, "test").loader()
            self.test_loop = LoopSelector(self.cfg["loop"]).get_test()

    def _initialize_awl(self):
        self.awl = AutomaticWeightedLoss(num_losses=3)
        self.awl.to(self.device)

    def train(self):
        """Main training loop."""
        try:
            save_json(self.cfg, self.path, "cfg.json")
            
            for epoch in range(self.start, self.end):
                self._train_epoch(epoch)
                #self._validate_epoch(epoch)
                #self._save_state(epoch)

                #if self.scheduler:
                #    self._handle_scheduler_step(epoch)
                garbage_collector()

        except Exception as e:
            logging.error(f"Error during training: {str(e)}")

    def _train_epoch(self, epoch):
        self.logger["epochs"].append(epoch)
        print("================================================================================")
        print(" [Epoch: %s]" %epoch)
        print("================================================================================")
        print(" --- Training ------------------------------------------------------------------")
        if self.cfg["loop"]["type"] == "multi_task":
            self.iter_count = self.train_loop(self.model, 
                self.train_loader,
                self.optimizer,
                self.scaler,
                self.logger,
                self.device,
                self.iter_count,
                epoch,
                self.awl)
        else:
            self.iter_count = self.train_loop(self.model, 
                self.train_loader,
                self.optimizer,
                self.scaler,
                self.logger,
                self.device,
                self.iter_count,
                epoch)

        garbage_collector()

    def _validate_epoch(self, epoch):
        print(" --- Validation ----------------------------------------------------------------")
        self.val_loop(self.model,
            self.val_loader,                
            self.scaler,
            self.logger,
            self.device,
            epoch,
            self.path
            )
        garbage_collector()

    def _save_state(self, epoch):
        save_model(epoch, 
            self.model, 
            self.optimizer, 
            self.path, 
            "last_model.pth")

        self.best_loss = best_loss_saver(epoch,
            self.logger, 
            self.model,
            self.optimizer,
            self.path,
            self.path_name,
            self.best_loss)
        
        """Just Keeping This Here For Now"""
        # HAD TO COMMENT OUT FOR CLASSIFICATION
        # TIDY THIS UP
        #if self.logger["mAP"][-1] > best_mAP:
        #    save_model(epoch, 
        #        self.model, 
        #        self.optimizer, 
        #        self.cfg["logging"]["path"], 
        #        "best_mAP.pth")
        #    best_mAP = self.logger["mAP"][-1]
        #    self.logger["best_mAP"].append(best_mAP)
        #    self.logger["best_mAP_epoch"].append(epoch) 

        save_json(self.logger,
            self.path,
            "log.json")

    def _handle_scheduler_step(self, epoch):
        if epoch == self.optimizer_cfg["scheduler"]["params"]["step"] -1:
            schedule_loader(self.model,
                self.path,
                self.path_name)
            self.best_loss = float('inf')
        self.scheduler.step()

    def test(self):
        """Testing loop."""
        self._test_model("best")

        # If step scheduler is used, test the pre-step model as well.
        if "scheduler" in self.optimizer_cfg:
            self._test_model("pre_step")

    def _test_model(self, state="best"):
        """Helper method for testing model in a given state (best/pre-step)."""
        filename = self.path_name if state == "best" else "/ps_" + self.path_name
        try:
            load_model(self.path_name, self.path_name, self.model)

            print("================================================================================")
            print(f" Evaluating {state.capitalize()} Model")
            print("================================================================================")

            self.test_loop(self.model, self.test_loader, self.device, self.path_name, train_flag=True)
            garbage_collector()
        
        except Exception as e:
            logging.error(f"Error during testing: {str(e)}")