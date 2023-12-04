"""
Module Detial:
    Implements a modified version of the mask r-cnn model. making the model
    a multi task model able to carry out both supervised instance segmentation
    and self supervised spatially representative classification
"""
# imports
# base packages
import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

# third party packages
import torch
import torch.nn as nn
import torchvision
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
    
    @torch.jit.unused
    def eager_outputs(self, losses, detections, rot_pred):
        if self.training:
            return losses, rot_pred

        return detections

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

    def forward(self, images, targets=None):
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training:
            if targets is None:
                
                # Handling just RotNet for SSL applications
                part_features = self.backbone.body(images)
                rot_features = part_features["3"]          
                rot_pred = self.self_supervised_head(rot_features)
                return rot_pred
            
            else:
                for target in targets:
                    boxes = target["boxes"]
                    if isinstance(boxes, torch.Tensor):
                        torch._assert(
                            len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                            f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                        )
                    else:
                        torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}.",
                    )

        part_features = self.backbone.body(images.tensors)
        features = self.backbone.fpn(part_features)
        
        # Handling RotNet for multi task execution
        rot_features = part_features["3"]
        rot_pred = self.self_supervised_head(rot_features)   

        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        #if torch.jit.is_scripting():
        #    if not self._has_warned:
        #        warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
        #        self._has_warned = True
        #    return losses, detections
        #else:
        #    return self.eager_outputs(losses, detections, rot_pred)
        if self.training:
            return losses, rot_pred

        return detections
                          
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