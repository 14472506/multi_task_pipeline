"""
Details
"""
# imports
from .mask_rcnn_resnet_50_fpn import Mask_RCNN_Resnet_50_FPN

# model builder
class ModelBuilder():
    def __init__(self, cfg):
        self.cfg = cfg

    def model_builder(self):
        """
        Details
        """
        # Mask RCNN Model with ResNet50 backbone with FPN
        if self.cfg["model_name"] == "Mask_RCNN_Resnet_50_FPN":
            model = Mask_RCNN_Resnet_50_FPN(self.cfg)
            return model

    
    