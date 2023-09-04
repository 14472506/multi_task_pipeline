"""
Detials
"""
# imports
import os
import json
import torch

# class
class Logs():
    """ Detials """
    def __init__(self, cfg):
        """ Detials """
        self.cfg = cfg
        self.root = "./outputs"
        self.logs_root = "logs"
        self.mod_root = "models"
        self.results_root = "results" 

        self._extract_config()
        self._make_dirs()
    
    def _extract_config(self):
        """ Detials """
        self.model_name = self.cfg["model_name"]
        self.logs_type = self.cfg["logger_type"]
        self.print_freq = self.cfg["print_freq"]
        self.dir = self.cfg["exp_dir"]
        self.sub_dir = self.cfg["sub_dir"]
        self.last_model_title = self.cfg["last_title"]
        self.post_model_title = self.cfg["best_post_title"]
        self.pre_model_title = self.cfg["best_pre_title"]
        self.iter = self.cfg["iter_init"]
        self.best = self.cfg["best_init"]
        self.step = self.cfg["step"]
    
    def _make_dirs(self):
        """ Detials """
        self.log_path = os.path.join(self.root, self.logs_root, self.dir, self.sub_dir)
        self.models_path = os.path.join(self.root, self.mod_root, self.dir, self.sub_dir)
        self.result_path = os.path.join(self.root, self.results_root, self.dir, self.sub_dir)

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        if not os.path.exists(self.models_path):
            os.makedirs(self.models_path)
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

    def get_iter(self):
        """ Detials """
        return self.iter
        
    def init_log_file(self, cfg, log):
        """ Detials """
        config_title = "config.json"
        config_root = os.path.join(self.log_path, config_title)
        with open(config_root, "w") as config_file:
            json.dump(cfg, config_file)

        log_title = "log.json"
        log_root = os.path.join(self.log_path, log_title)
        with open(log_root, "w") as log_file:
            json.dump(log, log_file)

    def update_log_file(self, log):
        """ Detials """
        log_title = "log.json"
        log_root = os.path.join(self.log_path, log_title)
        with open(log_root, "w") as log_file:
            json.dump(log, log_file)
    
    def get_log(self):
        """ Detials """
        logs_mapper = {
            "rotnet_resnet_50": self._class_logs,
            "rotmask_multi_task": self._multitask_logs
        }
        return logs_mapper[self.model_name]()
    
    def _class_logs(self):
        """ Details """
        if self.logs_type == "basic":
            logger = {
                "train_loss": [],
                "val_loss": [],
                "epochs": [],
                "pre_best_val": [],
                "pre_best_epoch": [],
                "post_best_val": [],
                "post_best_epoch": []
            }
        return logger
    
    def _multitask_logs(self):
        if self.logs_type == "basic":
            logger = {
                "train_loss": [],
                "train_sup_loss": [],
                "train_sup_ssl_loss": [],
                "train_ssl_loss": [],               
                "val_loss": [],
                "val_sup_loss": [],
                "val_sup_ssl_loss": [],
                "val_ssl_loss": [],
                "epochs": [],
                "pre_best_val": [],
                "pre_best_epoch": [],
                "post_best_val": [],
                "post_best_epoch": []
            }
        return logger
    
    def train_loop_reporter(self, epoch, iter_count, device, pf_loss):
        """ Detials """
        if iter_count % self.print_freq == 0: #self.print_freq-1:
            # get GPU memory usage
            loss = pf_loss/self.print_freq
            mem_all = torch.cuda.memory_allocated(device) / 1024**3 
            mem_res = torch.cuda.memory_reserved(device) / 1024**3 
            mem = mem_res + mem_all
            mem = round(mem, 2)
            print("[epoch: %s][iter: %s][memory use: %sGB] loss: %s" %(epoch ,iter_count, mem, loss))
            pf_loss = 0
            return pf_loss
        else:
            return pf_loss

    def val_loop_reporter(self, epoch, device, loss):
        """ Detials """
        # get GPU memory usage
        mem_all = torch.cuda.memory_allocated(device) / 1024**3 
        mem_res = torch.cuda.memory_reserved(device) / 1024**3 
        mem = mem_res + mem_all
        mem = round(mem, 2)
        print("[epoch: %s][iter: ---][memory use: %sGB] loss: %s" %(epoch, mem, loss))

    def save_model(self, epoch, model, optimiser, type = "last"):
        """ Detials """
        path_mapper = {
            "last": os.path.join(self.models_path, self.last_model_title),
            "pre": os.path.join(self.models_path, self.pre_model_title),
            "post": os.path.join(self.models_path, self.post_model_title)
        }
        checkpoint = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimiser.state_dict(),
        }
        torch.save(checkpoint, path_mapper[type])
    
    def load_model(self, model, type="last"):
        """ detials """
        path_mapper = {
            "last": os.path.join(self.models_path, self.last_model_title),
            "pre": os.path.join(self.models_path, self.pre_model_title),
            "post": os.path.join(self.models_path, self.post_model_title)
        }
        checkpoint = torch.load(path_mapper[type])
        model.load_state_dict(checkpoint["state_dict"])