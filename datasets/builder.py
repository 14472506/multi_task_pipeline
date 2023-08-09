import torch
import numpy as np
import random
from .COCODataset import COCODataset, COCO_collate_function
from .RotNetDataset import RotNetDataset
from .JigsawDataset import JigsawDataset
from .COCO_RotNet_Dataset import COCORotDataset, COCO_collate_function
from .augmentation_wrappers import RotNetWrapper, JigsawWrapper
from .augmentations import Augmentations

class DataloaderBuilder():
    """
    This class builds dataloaders based on given configurations and model requirements.
    """

    def __init__(self, cfg, load_type): 
        self.cfg = cfg
        self.dataset_cfg = cfg["dataset"]
        self.model_cfg = cfg["model"] 
        self.load_type = load_type
        self.augment = self.dataset_cfg["params"][self.load_type]["augment"]
        self.model_name = self.model_cfg["model_name"]
        self.seed = 42

    def loader(self):
        """
        Returns the appropriate dataloader based on the model configuration.
        """
        model_to_loader = {
            "Mask_RCNN_Resnet_50_FPN": self.coco_loader,
            "RotNet_ResNet_50": self.ssl_loader,  # RotNet and Jigsaw have similar structure.
            "Jigsaw_ResNet_50": self.ssl_loader,  # So, we can use a unified SSL loader.
            "Multi_task_RotNet_Mask_RCNN_Resnet50": self.rotnet_multitask_loader
        }

        return model_to_loader.get(self.model_name, self._unrecognized_model)()

    def _unrecognized_model(self):
        """
        Raises an error for unrecognized model names.
        """
        raise ValueError(f"Unrecognized model name '{self.model_name}'.")

    #def _split_dataset(self, dataset):
    #    """
    #    Splits a given dataset into train, validation, and test based on configurations.
    #    """
    #    # ... your splitting logic ...

    #def _init_fn_worker(self, seed):
    #    """
    #    Initializes random seeds for workers.
    #    """
    #    np.random.seed(seed)
    #    random.seed(seed)

    def _create_dataloader(self, dataset, collate_fn=None):
        """
        Utility function to create a DataLoader from a dataset.
        """
        gen = torch.Generator()
        gen.manual_seed(42)  # or any other value

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.dataset_cfg["params"][self.load_type]["batch_size"],
            shuffle=self.dataset_cfg["params"][self.load_type]["shuffle"],
            num_workers=self.dataset_cfg["params"][self.load_type]["num_workers"],
            worker_init_fn=lambda wid: self._init_fn_worker(42),  # Unique seed per worker.
            generator=gen,
            collate_fn=collate_fn
        )

    def coco_loader(self):
        """
        Creates a DataLoader for the COCO dataset.
        """
        
        def init_fn_worker(seed):
            np.random.seed(seed)
            random.seed(seed)

        gen = torch.Generator()
        gen.manual_seed(self.seed)

        cfg = self.dataset_cfg["params"][self.load_type]
        if self.augment:
            dataset = COCODataset(self.dataset_cfg["params"]["dir"], self.dataset_cfg["params"]["json_dir"], transforms=True, train=True)
        else:
            dataset = COCODataset(self.dataset_cfg["params"]["dir"], self.dataset_cfg["params"]["json_dir"])

        dataloader = self._create_dataloader(dataset, cfg, COCO_collate_function)
        return dataloader

    def ssl_loader(self):  # For RotNet and Jigsaw
        """
        Creates a DataLoader for Self-Supervised Learning datasets like RotNet and Jigsaw.
        """
        DatasetClass = RotNetDataset if self.model_name == "RotNet_ResNet_50" else JigsawDataset
        aug_type = "RotNet" if self.model_name == "RotNet_ResNet50" else "Jigsaw"                 # this is messy as fuck

        def init_fn_worker(seed):
            np.random.seed(seed)
            random.seed(seed)

        gen = torch.Generator()
        gen.manual_seed(self.seed)

        # get all data 
        all_data = DatasetClass(self.dataset_cfg, self.seed)

        # splitting data into train and test
        train_base_size = int(len(all_data)*self.dataset_cfg["splits"]["train_test"])
        test_size = len(all_data) - train_base_size
        train_base, test = torch.utils.data.random_split(all_data, [train_base_size, test_size]) 

        # just not doing this if not needed
        if "train" in self.cfg["loop"]["actions"]:
            # splitting train into train and val
            train_size = int(len(train_base)*self.dataset_cfg["splits"]["train_val"])
            validation_size = len(train_base) - train_size
            train, validation = torch.utils.data.random_split(train_base, [train_size, validation_size])
        
        # selecting only 
        if self.load_type == "test":
            dataset = test
        if self.load_type == "train":
            dataset = train
            if self.augment == True:
                Aug_loader = Augmentations(aug_type)
                Augs = Aug_loader.aug_loader()
                train == RotNetWrapper(train, Augs)
                print("augs applied")
        if self.load_type == "val":
            dataset = validation

        dataloader = self._create_dataloader(dataset)
        return dataloader

    def rotnet_multitask_loader(self):
        """
        Creates DataLoaders for multitask training with RotNet and COCO datasets.
        """
        # ... your logic ...
        ssl_dataloader = self._create_dataloader(dataset, cfg)
        coco_dataloader = self.coco_loader()
        return [coco_dataloader, ssl_dataloader]

"""
    def rotnet_multitask_loader(self, seed=42):

        def init_fn_worker(seed):
            np.random.seed(seed)
            random.seed(seed)

        gen = torch.Generator()
        gen.manual_seed(seed)

        # ----- RotNet loader for RotNet -------------------------------------------------------- #
        # get all data 
        all_data = RotNetDataset(self.cfg["dataset"]["rotnet"]["dir"], self.cfg["model"]["num_rotations"],)

        # splitting data into train and test
        train_base_size = int(len(all_data)*self.cfg["dataset"]["rotnet"]["train_test_split"])
        test_size = len(all_data) - train_base_size
        train_base, test = torch.utils.data.random_split(all_data, [train_base_size, test_size]) 

        # just not doing this if not needed
        if self.cfg["loop"]["train"]:
            # splitting train into train and val
            train_size = int(len(train_base)*self.cfg["dataset"]["rotnet"]["train_val_split"])
            validation_size = len(train_base) - train_size
            train, validation = torch.utils.data.random_split(train_base, [train_size, validation_size])
        
        # selecting only 
        if self.load_type == "test":
            dataset = test
        if self.load_type == "train":
            dataset = train
            if self.augment == True:
                Aug_loader = Augmentations("RotNet")
                Augs = Aug_loader.aug_loader()
                train == RotNetWrapper(train, Augs)
                print("augs applied")
        if self.load_type == "val":
            dataset = validation
        
        # getting specific loader config based on loader type
        cfg = self.cfg["dataset"]["rotnet"][self.load_type]
        
        ssl_dataloader = torch.utils.data.DataLoader(dataset,
            batch_size = cfg["batch_size"],
            shuffle = cfg["shuffle"],
            num_workers = cfg["num_workers"],
            worker_init_fn = init_fn_worker,
            generator = gen)

        # ----- COCO loader for Mask-RCNN ------------------------------------------------------- #
        cfg = self.cfg["dataset"]["mask_rcnn"][self.load_type]
        if cfg["augment"] == True:
            dataset = COCODataset(cfg["dir"], cfg["json_dir"], transforms=True, train=True)
        else:
            dataset = COCODataset(cfg["dir"], cfg["json_dir"])

        dataloader = torch.utils.data.DataLoader(dataset,
            batch_size = cfg["batch_size"],
            shuffle = cfg["shuffle"],
            num_workers = cfg["num_workers"],
            worker_init_fn = init_fn_worker,
            generator = gen,
            collate_fn = COCO_collate_function)

        loaders = [dataloader, ssl_dataloader]

        # returning loaders
        return loaders
"""