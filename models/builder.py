"""
Details
"""
# imports
from .mask_rcnn_resnet_50_fpn import Mask_RCNN_Resnet_50_FPN
from .rotnet_resnet_50 import RotNet_Resnet_50
from .multi_task_rot_rcnn_dev2 import Multi_task_RotNet_Mask_RCNN_Resnet_50_FPN
from .multi_task_rot_rcnn_dev3 import RotMaskRCNN_MultiTask
from .jigsaw_resnet_50 import Jigsaw_ResNet_50

# model build
class ModelBuilder():
    def __init__(self, cfg):
        self.cfg = cfg
        # Mapping model names to their respective classes
        self.model_mapping = {
            "Mask_RCNN_Resnet_50_FPN": Mask_RCNN_Resnet_50_FPN,
            "RotNet_ResNet_50": RotNet_Resnet_50,
            "Jigsaw_ResNet_50": Jigsaw_ResNet_50,
            "RotMaskRCNN_MultiTask": RotMaskRCNN_MultiTask
            # Add other model mappings here as you develop new models
        }

    def model_builder(self):
        """Builds and returns the model based on the configuration."""
        model_name = self.cfg["model_name"]
        
        # Fetch the model class based on the name and instantiate it
        if model_name in self.model_mapping:
            return self.model_mapping[model_name](self.cfg)
        else:
            raise ValueError(f"Model '{model_name}' is not recognized. Please check the model name or update the model mapping.")


























"""
Details
"""
# imports
from .mask_rcnn_resnet_50_fpn import Mask_RCNN_Resnet_50_FPN
from .rotnet_resnet_50 import RotNet_Resnet_50
from .multi_task_rot_rcnn_dev2 import Multi_task_RotNet_Mask_RCNN_Resnet_50_FPN
from .multi_task_rot_rcnn_dev3 import RotMaskRCNN_MultiTask
from .jigsaw_resnet_50 import Jigsaw_ResNet_50

# model build
class ModelBuilder():
    def __init__(self, cfg):
        self.cfg = cfg

    def model_builder(self):
        """
        Details
        """
        # ================================================
        # Instance Segmentation Models
        # ================================================
        # Mask RCNN Model with ResNet50 backbone with FPN
        if self.cfg["model_name"] == "Mask_RCNN_Resnet_50_FPN":
            model = Mask_RCNN_Resnet_50_FPN(self.cfg)
            return model

        # ================================================
        # Self Supervised Models
        # ================================================
        if self.cfg["model_name"] == "RotNet_ResNet_50":
            model = RotNet_Resnet_50(self.cfg)
            return model
        if self.cfg["model_name"] == "Jigsaw_ResNet_50":
            model = Jigsaw_ResNet_50(self.cfg)
            return model

        # ================================================
        # Combined Multi-task Models
        # ================================================
        if self.cfg["model_name"] == "RotMaskRCNN_MultiTask":
            model = RotMaskRCNN_MultiTask(self.cfg)
            return model

        

    
    