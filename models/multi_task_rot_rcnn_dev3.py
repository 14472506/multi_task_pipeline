"""
Detials
"""
# Imports
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from enum import Enum

class TaskType(Enum):
    SELF_SUPERVISED = "self_supervised"
    SUPERVISED = "supervised"

class RotMaskRCNN_MultiTask(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Backbone Initialization
        if cfg["backbone_type"] == "pre-trained":
            self.shared_backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
                backbone_name="resnet50",
                pretrained=True,  # This assumes torchvision's pretrained weights. Adjust as needed.
                trainable_layers=5)
        else:
            self.shared_backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
                backbone_name="resnet50",
                pretrained=False,
                trainable_layers=5)

        # RotNet and Mask R-CNN Heads
        self.rotnet_head = RotNetHead(self.shared_backbone.body)
        self.mask_rcnn_head = MaskRCNNHead(self.shared_backbone, cfg["num_classes"])

    def forward(self, x, y, task):
        if task == TaskType.SELF_SUPERVISED:
            return self.rotnet_head(x)
        elif task == TaskType.SUPERVISED:
            return self.mask_rcnn_head(x, y)
        else:
            raise ValueError(f"Invalid task type: {task}")

class RotNetHead(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_layers = nn.Sequential(
            nn.Linear(2048, 1000, bias=False))

        self.rotnet_classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1000, 4096, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1000, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 4))
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.avg_pooling(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_layers(x)
        x = self.rotnet_classifier(x)
        return x

class MaskRCNNHead(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()

        self.Mask_RCNN = torchvision.models.detection.MaskRCNN(
            backbone,
            num_classes=num_classes)

        # Configure the Mask R-CNN
        in_features = self.Mask_RCNN.roi_heads.box_predictor.cls_score.in_features
        self.Mask_RCNN.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        in_features_mask = self.Mask_RCNN.roi_heads.mask_predictor.conv5_mask.in_channels
        self.Mask_RCNN.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

    def forward(self, x, y=None):
        if y is None:
            return self.Mask_RCNN(x)
        else:
            return self.Mask_RCNN(x, y)

# === test === #
if __name__ == "__main__":
    cfg = {"backbone_type": "none", "num_classes": 2}
    test = RotMaskRCNN_MultiTask(cfg)
    print(test.rotnet_head)