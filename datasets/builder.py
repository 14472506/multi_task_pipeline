import torch
import numpy as np
import random
from .COCODataset import COCODataset, COCO_collate_function
from .RotNetDataset import RotNetDataset
from .JigsawDataset import JigsawDataset
from .COCO_RotNet_Dataset import COCORotDataset, COCO_collate_function
from .augmentation_wrappers import RotNetWrapper, JigsawWrapper, MultiTaskWrapper
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
        try:
            self.augment = self.dataset_cfg["params"][self.load_type]["augment"]
        except KeyError:
            pass
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
            "RotMaskRCNN_MultiTask": self.rotnet_multitask_loader
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
        # splitting data into train and test
        train_base_size = int(len(dataset)*self.dataset_cfg["params"]["splits"]["train_test"])
        test_size = len(dataset) - train_base_size
        train_base, test = torch.utils.data.random_split(dataset, [train_base_size, test_size]) 

        # just not doing this if not needed
        if "train" in self.cfg["loop"]["actions"]:
            # splitting train into train and val
            train_size = int(len(train_base)*self.dataset_cfg["params"]["splits"]["train_val"])
            validation_size = len(train_base) - train_size
            train, validation = torch.utils.data.random_split(train_base, [train_size, validation_size])
            return [test, train, validation]
        else:
            return [test]

    def _init_fn_worker(self, seed):
        """
        Initializes random seeds for workers.
        """
        np.random.seed(seed)
        random.seed(seed)

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
            worker_init_fn=self._init_fn_worker(42),  # Unique seed per worker.
            generator=gen,
            collate_fn=collate_fn
        )

    def coco_loader(self):
        """
        Creates a DataLoader for the COCO dataset.
        """
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

        gen = torch.Generator()
        gen.manual_seed(self.seed)

        # dataset
        all_data = DatasetClass(self.dataset_cfg, self.seed)
        splits = self._split_dataset(all_data)
        
        # selecting only 
        if self.load_type == "test":
            dataset = splits[0]
        if self.load_type == "train":
            dataset = splits[1]
            if self.augment == True:
                Aug_loader = Augmentations(aug_type)
                Augs = Aug_loader.aug_loader()
                dataset == RotNetWrapper(dataset, Augs)
                print("augs applied")
        if self.load_type == "val":
            dataset = splits[2]

        dataloader = self._create_dataloader(dataset)
        return dataloader

    def rotnet_multitask_loader(self):
        """
        Creates DataLoaders for multitask training with RotNet and COCO datasets.
        """
        gen = torch.Generator()
        gen.manual_seed(self.seed)        

        COCO_dataset = COCORotDataset(self.dataset_cfg, self.load_type, self.seed)

        if self.dataset_cfg["params"][self.load_type]["augment"]:
            COCO_Aug_loader = Augmentations("multi_task")
            COCO_Augs = COCO_Aug_loader.aug_loader()
            COCO_dataset = MultiTaskWrapper(COCO_dataset, COCO_Augs)
            print("augs applied")
        
        COCO_dataloader = self._create_dataloader(COCO_dataset, COCO_collate_function)

        # ROTNET STUFF
        self.dataset_cfg["params"]["train"]["batch_size"] = self.dataset_cfg["params"]["train"]["ssl_batch_size"]
        self.dataset_cfg["params"]["test"]["batch_size"] = self.dataset_cfg["params"]["test"]["ssl_batch_size"]
        self.dataset_cfg["params"]["val"]["batch_size"] = self.dataset_cfg["params"]["val"]["ssl_batch_size"]

        all_data = RotNetDataset(self.dataset_cfg, self.seed)
        splits = self._split_dataset(all_data)
        
        # selecting only 
        if self.load_type == "test":
            RotNet_dataset = splits[0]
        if self.load_type == "train":
            RotNet_dataset = splits[1]
            if self.augment == True:
                RotNet_Aug_loader = Augmentations("RotNet")
                RotNet_Augs = RotNet_Aug_loader.aug_loader()
                RotNet_dataset == RotNetWrapper(RotNet_dataset, RotNet_Augs)
                print("augs applied")
        if self.load_type == "val":
            RotNet_dataset = splits[2]

        RotNet_dataloader = self._create_dataloader(RotNet_dataset)

        return [COCO_dataloader, RotNet_dataloader]