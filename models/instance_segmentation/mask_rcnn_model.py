"""
Mask R-CNN baseline Model
"""
# imports
import torch
import torch.nn as nn
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator

def maskrcnn_resnet50_fpn(cfg):
                         
    # confing argument 
    backbone_type = cfg["params"]["backbone_type"]
    trainable_layers = cfg["params"]["trainable_layers"]
    num_classes = cfg["params"]["num_classes"]
    hidden_layers = cfg["params"]["hidden_layers"]

    # backbone selecting
    if backbone_type == "pre-trained":
        backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
                    backbone_name="resnet50",
                    weights="ResNet50_Weights.DEFAULT",
                    trainable_layers=trainable_layers)
    else:
        backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
                    backbone_name="resnet50",
                    weights=False,
                    trainable_layers=trainable_layers)

    model = torchvision.models.detection.MaskRCNN(backbone, num_classes)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layers, num_classes)
    
    return model