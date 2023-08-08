import torch
import numpy as np
import random
# ... other imports ...

class DataloaderBuilder():
    """
    This class builds dataloaders based on given configurations and model requirements.
    """

    def __init__(self, cfg, load_type):
        self.cfg = cfg
        self.load_type = load_type
        self.augment = self.cfg["dataset"][self.load_type]["augment"]
        self.model_name = self.cfg["model"]["model_name"]

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

    def _split_dataset(self, dataset):
        """
        Splits a given dataset into train, validation, and test based on configurations.
        """
        # ... your splitting logic ...

    def _init_fn_worker(self, seed):
        """
        Initializes random seeds for workers.
        """
        np.random.seed(seed)
        random.seed(seed)

    def _create_dataloader(self, dataset, cfg, collate_fn=None):
        """
        Utility function to create a DataLoader from a dataset.
        """
        gen = torch.Generator()
        gen.manual_seed(42)  # or any other value

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            shuffle=cfg["shuffle"],
            num_workers=cfg["num_workers"],
            worker_init_fn=lambda wid: self._init_fn_worker(42 + wid),  # Unique seed per worker.
            generator=gen,
            collate_fn=collate_fn
        )

    def coco_loader(self):
        """
        Creates a DataLoader for the COCO dataset.
        """
        # ... your logic ...
        dataloader = self._create_dataloader(dataset, cfg, COCO_collate_function)
        return dataloader

    def ssl_loader(self):  # For RotNet and Jigsaw
        """
        Creates a DataLoader for Self-Supervised Learning datasets like RotNet and Jigsaw.
        """
        DatasetClass = RotNetDataset if self.model_name == "RotNet_ResNet_50" else JigsawDataset
        # ... your logic ...
        dataloader = self._create_dataloader(dataset, cfg)
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
Details
"""
# import
import torch
import numpy as np
import random
from .COCODataset import COCODataset, COCO_collate_function
from .RotNetDataset import RotNetDataset
from .JigsawDataset import JigsawDataset
from .COCO_RotNet_Dataset import COCORotDataset, COCO_collate_function
from .augmentation_wrappers import RotNetWrapper, JigsawWrapper
from .augmentations import Augmentations

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
        self.augment = self.cfg["dataset"][self.load_type]["augment"]
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
        if self.cfg["model"]["model_name"] == "Jigsaw_ResNet_50":
            loader = self.jigsaw_loader()
            return loader
        if self.cfg["model"]["model_name"] == "Multi_task_RotNet_Mask_RCNN_Resnet50":
            loader = self.rotnet_multitask_loader()
            return loader
    
    def coco_loader(self, seed=42):
        """
        Detials
        """
        def init_fn_worker(seed):
            np.random.seed(seed)
            random.seed(seed)

        gen = torch.Generator()
        gen.manual_seed(seed)

        cfg = self.cfg["dataset"][self.load_type]
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
        
        return dataloader

    # LOT OF COPY AND PAST CODE BETWEEN ROTNET AND JIGSAW DLS, THIS SHOULD BE ADDRESSED. PS AND JIGROT?
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
            if self.augment == True:
                Aug_loader = Augmentations("RotNet")
                Augs = Aug_loader.aug_loader()
                train == RotNetWrapper(train, Augs)
                print("augs applied")
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
    
    def jigsaw_loader(self, seed=42):
        """
        Details
        """
        def init_fn_worker(seed):
            np.random.seed(seed)
            random.seed(seed)

        # get all data 
        all_data = JigsawDataset(self.cfg["dataset"]["dir"], self.cfg["model"]["num_tiles"], self.cfg["model"]["num_permutations"])

        gen = torch.Generator()
        gen.manual_seed(seed)

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
            if self.augment == True:
                Aug_loader = Augmentations("Jigsaw")
                Augs = Aug_loader.aug_loader()
                train == JigsawWrapper(train, Augs)
                print("augs applied")
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

    def rotnet_multitask_loader(self, seed=42):
        """
        Details
        """
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