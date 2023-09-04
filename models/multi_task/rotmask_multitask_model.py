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

# class
class RotMask_Multi_Task(nn.Module):
    """ Detials """
    def __init__(self, cfg):
        """ Detials """
        super().__init__()
        self.cfg = cfg
        self._extract_config()

        # Backbone Definition
        self._get_backbone()

        # Mask R-CNN Model Definition
        self.Mask_RCNN = torchvision.models.detection.MaskRCNN(
            self.backbone,
            num_classes=self.num_classes)
        in_features = self.Mask_RCNN.roi_heads.box_predictor.cls_score.in_features
        self.Mask_RCNN.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        in_features_mask = self.Mask_RCNN.roi_heads.mask_predictor.conv5_mask.in_channels
        self.Mask_RCNN.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, self.hidden_layers, self.num_classes)

        # RotNet Classifier Head Definition
        self.avg_pooling = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))
        self.fc_layers = nn.Sequential(nn.Linear(2048, 1000, bias=False))
        self.classifier = nn.Sequential(
            nn.Dropout() if self.drop_out > 0. else nn.Identity(),
            nn.Linear(1000, 4096, bias=False if self.batch_norm else True),
            nn.BatchNorm1d(4096) if self.batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout() if self.drop_out > 0. else nn.Identity(),
            nn.Linear(4096, 4096, bias=False if self.batch_norm else True),
            nn.BatchNorm1d(4096) if self.batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout() if self.drop_out > 0. else nn.Identity(),
            nn.Linear(4096, 1000, bias=False if self.batch_norm else True),
            nn.BatchNorm1d(1000) if self.batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Linear(1000, self.num_rots))
        self.classifier = nn.Sequential(*[child for child in self.classifier.children() if not isinstance(child, nn.Identity)])

    def forward(self, x, y=None, action="supervised"):
        """ Details """
        if action == "supervised":
            x_hat = self.Mask_RCNN(x, y)
            return x_hat
        else:
            x_hat = self.Mask_RCNN.backbone.body(x)
            x_hat = x_hat["3"]
            x_hat = self.avg_pooling(x_hat)
            x_hat = torch.flatten(x_hat, start_dim=1)
            x_hat = self.fc_layers(x_hat)
            x_hat = self.classifier(x_hat)
            return x_hat

    def _extract_config(self):
        """ Detials """
        self.model_name = self.cfg["model_name"]
        self.backbone_type = self.cfg["params"]["backbone_type"]
        self.num_rots = self.cfg["params"]["num_rotations"]
        self.drop_out = self.cfg["params"]["drop_out"]
        self.batch_norm = self.cfg["params"]["batch_norm"]
        self.trainable_layers = self.cfg["params"]["trainable_layers"]
        self.num_classes = self.cfg["params"]["num_classes"]
        self.hidden_layers = self.cfg["params"]["hidden_layers"]
    
    def _get_backbone(self):
        """ Detials """
                # backbone selecting
        if self.backbone_type == "pre-trained":
            self.backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
                         backbone_name="resnet50",
                         weights="ResNet50_Weights.DEFAULT",
                         trainable_layers=self.trainable_layers)
        else:
            self.backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
                backbone_name="resnet50",
                weights=False,
                trainable_layers=self.trainable_layers)