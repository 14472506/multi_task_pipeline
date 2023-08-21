import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

class RotMaskRCNN_MultiTask(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Backbone Initialization
        self.shared_backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
            backbone_name="resnet50",
            pretrained=True,  # Adjust this as needed.
            trainable_layers=5)

        # MaskRCNN
        self.Mask_RCNN = torchvision.models.detection.MaskRCNN(self.shared_backbone, num_classes=2)
        in_features = self.Mask_RCNN.roi_heads.box_predictor.cls_score.in_features
        self.Mask_RCNN.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        in_features_mask = self.Mask_RCNN.roi_heads.mask_predictor.conv5_mask.in_channels
        self.Mask_RCNN.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, 2)

        # RotNet
        self.rotnet_backbone = self.shared_backbone.body
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

    def forward(self, x, y=None, task=None):
        if task == "self_supervised":
            x = self.rotnet_backbone(x)
            x = x["3"]
            x = self.avg_pooling(x)
            x = torch.flatten(x, start_dim=1)
            x = self.fc_layers(x)
            x = self.rotnet_classifier(x)
            return x
        else:
            if y == None:
                x = self.Mask_RCNN(x)
                return x
            else:
                x = self.Mask_RCNN(x, y)
                return x