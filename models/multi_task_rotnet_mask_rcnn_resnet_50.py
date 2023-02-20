"""
Detials
"""
# imports
import torch
import torch.nn as nn
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator

# model
class Multi_task_RotNet_Mask_RCNN_Resnet_50_FPN(nn.Module):
    """
    Detials
    """
    def __init__(self, cfg):
        super().__init__()
        # Backbone
        if cfg["backbone_type"] == "pre-trained":
            self.mask_rcnn_backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
                backbone_name="resnet50",
                weights="ResNet50_Weights.DEFAULT",
                trainable_layers=5)
        else:
            self.mask_rcnn_backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
                backbone_name="resnet50",
                weights=False,
                trainable_layers=5)
        # ad if else here for loading other kind of backbone weights

        self.rotnet_backbone = torchvision.models.resnet50(
                weights="ResNet50_Weights.DEFAULT")

        self.rotnet_backbone.load_state_dict(self.mask_rcnn_backbone.body.state_dict(), strict=False)

        # Mask R-CNN
        self.Mask_RCNN = Mask_RCNN(self.mask_rcnn_backbone, 2)

        # RotNet
        self.RotNet = RotNet(self.rotnet_backbone)
        
    def forward(self, rot_x, mask_x=None, target=None):
        """
        Detials
        """
        if mask_x == None:
            x = self.RotNet(rot_x)
            return x
        else:
            x1 = self.RotNet(rot_x)
            x2 = self.Mask_RCNN(mask_x, target)
            return x1, x2 

class Mask_RCNN(nn.Module):
    """
    Detials
    """
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.Mask_RCNN = torchvision.models.detection.MaskRCNN(
            backbone,
            num_classes=num_classes)

        # get number of input features for the classifier
        in_features = self.Mask_RCNN.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.Mask_RCNN.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        # now get the number of input features for the mask classifier
        in_features_mask = self.Mask_RCNN.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        self.Mask_RCNN.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                           hidden_layer,
                                                           num_classes)  
    
    def forward(self, x, y=None):
        if y == None:
            x = self.Mask_RCNN(x)
            return x
        else:
            x = self.Mask_RCNN(x, y)
            return x

class RotNet(nn.Module):
    """
    Detials
    """
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.rotnet_classifier = nn.Sequential(
            nn.Dropout() if 0.5 > 0. else nn.Identity(),
            nn.Linear(1000, 4096, bias=False),
            #nn.BatchNorm1d(4096) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout() if 0.5 > 0. else nn.Identity(),
            nn.Linear(4096, 4096, bias=False),
            #nn.BatchNorm1d(4096) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout() if 0.5 > 0. else nn.Identity(),
            nn.Linear(4096, 1000, bias=False),
            #nn.BatchNorm1d(1000) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 4))
        
        self.rotnet = nn.Sequential(
            self.backbone,
            self.rotnet_classifier)

    def forward(self, x):
        x = self.rotnet(x)
        return x

# test

