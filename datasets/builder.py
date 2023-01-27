"""
Details
"""
# import
import torch
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
    
    def coco_loader(self):
        """
        Detials
        """
        dataset = COCODataset(self.cfg["dir"], self.cfg["json_dir"])
        dataloader = torch.utils.data.DataLoader(dataset,
            batch_size = self.cfg["batch_size"],
            shuffle = self.cfg["shuffle"],
            num_workers = self.cfg["num_workers"],
            collate_fn = COCO_collate_function)
        return dataloader
