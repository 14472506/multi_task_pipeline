"""
Details
"""
# import
import torch
import numpy as np
import random
from .COCODataset import COCODataset, COCO_collate_function
from .RotNetDataset import RotNetDataset

# class
class DataloaderBuilder():
    """
    Details
    """
    def __init__(self, cfg, load_type):
        """
        Details
        """
        self.cfg = cfg
        self.load_type = load_type
        print(load_type)

    def loader(self):
        """
        Details
        """
        if self.cfg["model"]["model_name"] == "Mask_RCNN_Resnet_50_FPN":
            loader = self.coco_loader()
            return loader
        if self.cfg["model"]["model_name"] == "RotNet_ResNet_50":
            loader = self.rotnet_loader()
            return loader
    
    def coco_loader(self, seed=42):
        """
        Detials
        """
        cfg = self.cfg["dataset"][self.load_type]
        def init_fn_worker(seed):
            np.random.seed(seed)
            random.seed(seed)

        gen = torch.Generator()
        gen.manual_seed(seed)

        dataset = COCODataset(cfg["dir"], cfg["json_dir"])
        dataloader = torch.utils.data.DataLoader(dataset,
            batch_size = cfg["batch_size"],
            shuffle = cfg["shuffle"],
            num_workers = cfg["num_workers"],
            worker_init_fn = init_fn_worker,
            generator = gen,
            collate_fn = COCO_collate_function)
        return dataloader

    def rotnet_loader(self, seed=42):
        """
        Details
        """
        def init_fn_worker(seed):
            np.random.seed(seed)
            random.seed(seed)

        gen = torch.Generator()
        gen.manual_seed(seed)

        # get all data 
        all_data = RotNetDataset(self.cfg["dataset"]["dir"], self.cfg["model"]["num_rotations"],)

        # splitting data into train and test
        train_base_size = int(len(all_data)*self.cfg["dataset"]["train_test_split"])
        test_size = len(all_data) - train_base_size
        train_base, test = torch.utils.data.random_split(all_data, [train_base_size, test_size]) 

        # just not doing this if not needed
        if self.cfg["loop"]["train"]:
            # splitting train into train and val
            train_size = int(len(train_base)*self.cfg["dataset"]["train_val_split"])
            validation_size = len(train_base) - train_size
            train, validation = torch.utils.data.random_split(train_base, [train_size, validation_size])
        
        # selecting only 
        if self.load_type == "test":
            dataset = test
        if self.load_type == "train":
            dataset = train
        if self.load_type == "val":
            dataset = validation
        
        # getting specific loader config based on loader type
        cfg = self.cfg["dataset"][self.load_type]
        
        dataloader = torch.utils.data.DataLoader(dataset,
            batch_size = cfg["batch_size"],
            shuffle = cfg["shuffle"],
            num_workers = cfg["num_workers"],
            worker_init_fn = init_fn_worker,
            generator = gen)
        return dataloader