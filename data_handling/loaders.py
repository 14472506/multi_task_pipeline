"""
Detials
"""
# imports
from .datasets.rotnet import RotNetDataset#
from .transforms.transforms import Transforms
from .transforms.transform_wrapper import wrappers
import torch
import numpy as np 
import random

# class
class Loaders():
    """ Detials """
    def __init__(self, cfg, type=None):
        """ Detials """
        self.cfg = cfg
        self.type = type
        self._extract_config()
        self._get_dataset_class()

    def _extract_config(self):
        """ Detials """
        self.model_type = self.cfg["params"]["model_name"]
        self.random_seed = self.cfg["random_seed"]
        self.train_test_split = self.cfg["params"]["split"]["train_test"]
        self.train_val_split = self.cfg["params"]["split"]["train_val"]

        if self.type == "train":

            self.train_bs = self.cfg["params"]["train"]["batch_size"]
            self.val_bs = self.cfg["params"]["val"]["batch_size"]
            self.train_shuffle = self.cfg["params"]["train"]["shuffle"]
            self.val_shuffle = self.cfg["params"]["val"]["shuffle"]
            self.train_workers = self.cfg["params"]["train"]["num_workers"]
            self.val_workers = self.cfg["params"]["val"]["num_workers"]
            self.train_augs = self.cfg["params"]["train"]["augmentations"]
            self.val_augs = self.cfg["params"]["val"]["augmentations"]

        else:

            self.test_bs = self.cfg["params"]["test"]["batch_size"]
            self.test_shuffle = self.cfg["params"]["test"]["shuffle"]
            self.test_workers = self.cfg["params"]["test"]["num_workers"]
            self.test_augs = self.cfg["params"]["test"]["augmentations"]
        
        self.col_fn = self.cfg["params"]["col_fn"]


    def _get_dataset_class(self):
        """ Details """
        dataset_selector = {
            "rotnet_resnet_50": RotNetDataset
        }
        self.dataset_class = dataset_selector[self.model_type]

    def loader(self):
        """ Detials """
        loader_selector = {
            "rotnet_resnet_50": self._classifier_loader()
        }
        if self.type == "train":
            train_loader, val_loader = loader_selector[self.model_type]
            return train_loader, val_loader
        else:
            test_loader = loader_selector[self.model_type]
            return test_loader

    def _classifier_loader(self):  # For RotNet and Jigsaw
        """ Creates a DataLoader for Self-Supervised Learning datasets like RotNet and Jigsaw."""
        # dataset
        all_data = self.dataset_class(self.cfg, self.cfg["random_seed"])
        splits = self._data_split(all_data)
        
        if self.type == "test":

            test_dataset = splits[0]
            if self.test_augs:
                transforms = Transforms(self.cfg)
                transform_wrapper = wrappers(self.model_type)
                test_dataset = transform_wrapper(test_dataset, transforms)
                print("test augs applied")
            test_loader = self._create_dataloader(test_dataset, self.test_bs, self.test_shuffle, self.test_workers, self.col_fn)
            return test_loader
        
        if self.type == "train":
        
            train_dataset = splits[0]
            val_dataset = splits[1]

            if self.train_augs:
                transforms = Transforms(self.cfg).transforms()
                transform_wrapper = wrappers(self.model_type)
                train_dataset = transform_wrapper(train_dataset, transforms)
                print("train augs applied")

            if self.val_augs:
                transforms = Transforms(self.cfg)
                transform_wrapper = wrappers(self.model_type)
                val_dataset = transform_wrapper(val_dataset, transforms)
                print("val augs applied")

            train_loader = self._create_dataloader(train_dataset, self.train_bs, self.train_shuffle, self.train_workers, self.col_fn)
            val_loader = self._create_dataloader(val_dataset, self.val_bs, self.val_shuffle, self.val_workers, self.col_fn)
            return train_loader, val_loader

    def _data_split(self, dataset):
        """ Detials """
        # splitting data into train and test
        train_base_size = int(len(dataset)*self.train_test_split)
        test_size = len(dataset) - train_base_size
        train_base, test = torch.utils.data.random_split(dataset, [train_base_size, test_size]) 

        # just not doing this if not needed
        if self.type == "train":
            train_size = int(len(train_base)*self.train_val_split)
            validation_size = len(train_base) - train_size
            train, validation = torch.utils.data.random_split(train_base, [train_size, validation_size])
            return [train, validation]
        else:
            return [test]

    def _init_fn_worker(self, seed):
        """ Initializes random seeds for workers """
        np.random.seed(seed)
        random.seed(seed)

    def _create_dataloader(self, dataset, batch_size, shuffle, num_workers, collate_fn=None):
        """ Utility function to create a DataLoader from a dataset """
        gen = torch.Generator()
        gen.manual_seed(self.random_seed)  # or any other value

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            worker_init_fn=self._init_fn_worker(self.random_seed),  # Unique seed per worker.
            generator=gen,
            collate_fn=collate_fn
        )