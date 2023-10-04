"""
Module Detial:
    Implements a modified version of the mask r-cnn model. making the model
    a multi task model able to carry out both supervised instance segmentation
    and self supervised spatially representative classification
"""
# imports
# base packages

# third party packages
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# local packages

# classes
class RotNetHead(torch.nn.Module):
    """
    RotNetHead to be added to MaskRCNN class. the feature extractor is alread in the MaskRCNN class
    Which the RotMaskRCNN class will inherits from. The rotnet head in this class takes the
    features provided from the RotMaskRCNN model and returns the classificiation.  
    """
    def __init__(self, 
                 num_rots=4,
                 batch_norm=True,
                 drop_out=0.5):
        super().__init__()

        self.avg_pooling = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))
        self.fc_layers = nn.Sequential(nn.Linear(2048, 1000, bias=False))
        self.classifier = nn.Sequential(
            nn.Dropout() if drop_out > 0. else nn.Identity(),
            nn.Linear(1000, 4096, bias=False if batch_norm else True),
            nn.BatchNorm1d(4096) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout() if drop_out > 0. else nn.Identity(),
            nn.Linear(4096, 4096, bias=False if batch_norm else True),
            nn.BatchNorm1d(4096) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout() if drop_out > 0. else nn.Identity(),
            nn.Linear(4096, 1000, bias=False if batch_norm else True),
            nn.BatchNorm1d(1000) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Linear(1000, num_rots))
        self.classifier = nn.Sequential(*[child for child in self.classifier.children() if not isinstance(child, nn.Identity)])

    def forward(self, x):
        """ Details """ 
        x = self.avg_pooling(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_layers(x)
        x = self.classifier(x)
        return x

class RotMaskRCNN(torchvision.models.detection.MaskRCNN):
    def __init__(self, backbone=None, num_classes=91, num_rots=4, batch_norm=True, drop_out=0.5, **kwargs):
        # >>> potential further definitions here
        super().__init__(backbone=backbone, num_classes=num_classes, **kwargs)
        # above may need significantly more configuration, actually might be better
        # to include this anyway, then this can be place in the config for further
        # optimisation. for no provide just essential parts

        # aditional classifier head defined. not to be passed into roi, or should 
        # this be? <<< try this later, for now, backbone to classifier head. 
        self.self_supervised_head = RotNetHead(num_rots, batch_norm, drop_out)

    def forward(self, images, targets=None, mode='segm'):
        if mode == 'segm':
            # carry out normal forward as MRCNN
            return super().forward(images, targets)
        
        elif mode == 'ssl':
            # In 'rotnet' mode, we use only the RotNet head
            #features = self.backbone(images.tensors if isinstance(images, ImageList) else images)
            features = self.backbone.body(images)
            # select output to pass into model. This may need to change when tested
            in_features = features["3"]
            # classifier head
            x = self.self_supervised_head(in_features)

            return x
                  
# model init may need to go here too, how does the MRCNN or faster RCNN class do this?
def rotmask_resnet50_fpn(cfg):
                         
    # confing argument 
    backbone_type = cfg["params"]["backbone_type"]
    num_rots = cfg["params"]["num_rotations"]
    drop_out = cfg["params"]["drop_out"]
    batch_norm = cfg["params"]["batch_norm"]
    trainable_layers = cfg["params"]["trainable_layers"]
    num_classes = cfg["params"]["num_classes"]
    hidden_layers = cfg["params"]["hidden_layers"]
    min_size = cfg["params"]["min_size"]
    max_size = cfg["params"]["max_size"]

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
    
    if drop_out:
        backbone.body.layer4.add_module("dropout", nn.Dropout(drop_out))

    model = RotMaskRCNN(backbone, num_classes, num_rots, batch_norm, drop_out, max_size=max_size, min_size=min_size)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layers, num_classes)
    
    return model
