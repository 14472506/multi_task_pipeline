"""
Details
"""
# imports
from .mask_rcnn_resnet_50_fpn import Mask_RCNN_Resnet_50_FPN
from .rotnet_resnet_50 import RotNet_Resnet_50

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

        # ================================================
        # Combined Multi-task Models
        # ================================================

        

    
    