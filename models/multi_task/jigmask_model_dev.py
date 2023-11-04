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
class JigsawHead(torch.nn.Module):
    """
    JigsawHead to be added to MaskRCNN class. the feature extractor is alread in the MaskRCNN class
    Which the JigMaskRCNN class will inherits from. The rotnet head in this class takes the
    features provided from the JigMaskRCNN model and returns the classificiation.  
    """
    def __init__(self, num_tiles, num_permutations):
        """ Detials """
        super().__init__()
        self.num_tiles = num_tiles
        self.num_permutations = num_permutations

        self.twin_network = nn.Sequential(nn.Linear(1000, 512, bias=False),
                                          #nn.BatchNorm1d(512),
                                          nn.ReLU(inplace=True)
                                          )
        
        self.classifier = nn.Sequential(nn.Linear(512*self.num_tiles, 4096, bias=False),
                                         #nn.BatchNorm1d(4096),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(4096, self.num_permutations))

    def forward(self, x):
        """ Details """
        x = torch.stack([self.twin_network(param) for param in x])
        x = x.permute(1, 0, 2)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x
    
class JigMaskRCNN(torchvision.models.detection.MaskRCNN):
    def __init__(self, backbone=None, num_classes=91, num_tiles=9, num_permutations=100, batch_norm=True, drop_out=0.5, **kwargs):
        # >>> potential further definitions here
        super().__init__(backbone=backbone, num_classes=num_classes, **kwargs)
        # above may need significantly more configuration, actually might be better
        # to include this anyway, then this can be place in the config for further
        # optimisation. for no provide just essential parts

        # aditional classifier head defined. not to be passed into roi, or should 
        # this be? <<< try this later, for now, backbone to classifier head. 
        self.num_tiles = num_tiles

        self.jig_avg_pooling = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))
        self.jig_fc_layers = nn.Sequential(nn.Linear(2048, 1000, bias=False))
        self.self_supervised_head = JigsawHead(num_tiles, num_permutations)

    def forward(self, images=None, targets=None):
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
            if images.shape(1) == self.num_tiles:
                if targets is None:
                    # How to make this a method? does it need to be?
                    feature_stack = []
                    for i in range(self.num_tiles):
                        part_features = self.backbone.body(images[:, i, :, :, :])
                        part_features = self.jig_avg_pooling(part_features["3"])
                        part_features = self.jig_fc_layers(torch.flatten(part_features, start_dim=1))
                        feature_stack.append(part_features)
                    jig_features = torch.stack(feature_stack) 
                    jig_pred = self.self_supervised_head(jig_features)
                    return jig_pred
                
                else:
                    for image, target in zip(images, targets):
                        for i in range(self.num_tiles):
                            tile_im = image[:, i, :, :, :]
                            tile_mask = target["masks"]
            # #################################################################################
            # summat summat summat
            # #################################################################################            
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
        
        # Mask R-CNN Features backbone and fpn
        features = self.backbone(images.tensors)
        # Handling Jigsaw for multi task execution
        feature_stack = []
        for i in range(self.num_tiles):
            part_features = self.backbone.body(im_stack[i])
            part_features = self.jig_avg_pooling(part_features["3"])
            part_features = self.jig_fc_layers(torch.flatten(part_features, start_dim=1))
            feature_stack.append(part_features)
        jig_features = torch.stack(feature_stack)    
        jig_pred = self.self_supervised_head(jig_features)
 
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
            return losses, jig_pred

        return detections
                          
# model init may need to go here too, how does the MRCNN or faster RCNN class do this?
def jigmask_resnet50_fpn(cfg):
                         
    # confing argument 
    backbone_type = cfg["params"]["backbone_type"]
    num_tiles = cfg["params"]["num_tiles"]
    num_permutations = cfg["params"]["num_permutations"]
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

    model = JigMaskRCNN(backbone, num_classes, num_tiles, num_permutations, batch_norm, drop_out, max_size=max_size, min_size=min_size)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layers, num_classes)
    
    return model