"""
Details
"""
# import
import torch
import numpy as np
import random
from .COCODataset import COCODataset, COCO_collate_function

# class
class DataloaderBuilder():
    """
    Details
    """
    def __init__(self, cfg):
        """
        Details
        """
        self.cfg = cfg
    
    def coco_loader(self, seed=42):
        """
        Detials
        """

        def init_fn_worker(seed):
            np.random.seed(seed)
            random.seed(seed)

        gen = torch.Generator()
        gen.manual_seed(seed)

        dataset = COCODataset(self.cfg["dir"], self.cfg["json_dir"])
        dataloader = torch.utils.data.DataLoader(dataset,
            batch_size = self.cfg["batch_size"],
            shuffle = self.cfg["shuffle"],
            num_workers = self.cfg["num_workers"],
            worker_init_fn = init_fn_worker,
            generator = gen,
            collate_fn = COCO_collate_function)
        return dataloader
