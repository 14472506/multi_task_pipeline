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
from torch import Tensor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.image_list import ImageList

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
        
        self.classifier = nn.Sequential(nn.Linear(1000*self.num_tiles, 4096, bias=False),
                                         nn.BatchNorm1d(4096),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(p=0.5),
                                         nn.Linear(4096, 2048, bias=False),
                                         nn.BatchNorm1d(2048),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(p=0.5),
                                         nn.Linear(2048, 1024, bias=False),
                                         nn.BatchNorm1d(1024),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(p=0.5),
                                         nn.Linear(1024, self.num_permutations))

    def forward(self, x):
        """ Details """

        #x = x.permute(1, 0, 2)
        x = torch.flatten(x, start_dim=1)

        if x.size(0) == 1:
            for module in self.classifier:
                if isinstance(module, nn.BatchNorm1d):
                    x = module.eval()(x)
                else:
                    x = module(x)
        else:
            x = self.classifier(x)

        return x

class JigMaskRCNN(torchvision.models.detection.MaskRCNN):
    """
    JigMask is built of torch visions implemenetation of mask-rcnn, the class adds the jigrot 
    classification head alone with a modified forward to allow the tiles to be processed for both
    images and masks, along with the jigsaw classification
    """
    def __init__(self, backbone=None, num_classes=91, num_tiles=9, num_permutations=100, tile_min=266, tile_max=444, **kwargs):
        # >>> potential further definitions here
        super().__init__(backbone=backbone, num_classes=num_classes, **kwargs)
        # above may need significantly more configuration, actually might be better
        # to include this anyway, then this can be place in the config for further
        # optimisation. for no provide just essential parts

        self.num_tiles = num_tiles

        self.jig_avg_pooling = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))
        self.jig_fc_layers = nn.Sequential(nn.Linear(2048, 1000, bias=False))
        self.self_supervised_head = JigsawHead(num_tiles, num_permutations)

        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]

        self.tile_transform = GeneralizedRCNNTransform(tile_min, tile_max, image_mean, image_std)


    def forward(self, images=None, targets=None):
        """
        Args:
            images (list[Tensor]): image
                    ext_inputs = {
                        "tile_imgs": [],
                        "tile_masks": [],
                        "tile_boxes": [],
                        "tile_labels": [],
                    }s to be processed
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        # code for training execution
        if self.training:

            # if dimension is 4 and first dimension is the same as the number of tiles then it shoulds be a tile image and needs to be unsqueezer
            if images.dim() == 4:
                if images.shape[0] == self.num_tiles:
                    images = images.unsqueeze(0)

            # if an images an standard images and is of batch size one it must be unsqueezed to be processed in the following code
            if images.dim() == 3:
                images = images.unsqueeze(0)

            # init data collection for tiles
            jig_pred_list = []
            acc_losses = {
                "loss_classifier": 0,
                "loss_box_reg": 0,
                "loss_mask": 0,
                "loss_objectness": 0,
                "loss_rpn_box_reg": 0
            }

            # init acc loss count for tiles
            acc_loss_count = 0

            ## images will be of 5 dimensions with the first being of the batch size. this must be iterated through
            #for batch_idx in range(images.shape[0]):
            #    # select the image for the batch index
            #    image = images[batch_idx]
                
                # if images is tiled image.
                #if image.shape[1] == self.num_tiles:
            
            
            if targets is None:
                # TODO Make seperate method
                device = images.device
                images = list(image.to(device) for image in images)

                stack_1 = []
                for image in images:
                    stack_2 = []
                    for tile in image:
                        tile, _ = self.tile_transform([tile])
                        stack_2.append(tile.tensors.squeeze())
                    image_tensor = torch.stack(stack_2)
                    stack_1.append(image_tensor)
                images_tensor = torch.stack(stack_1).to(device)

                part_features = torch.stack([self.jig_fc_layers(torch.flatten(self.jig_avg_pooling(self.backbone.body(tile)["3"]), start_dim=1)) for tile in images_tensor])
                jig_pred_tensor = self.self_supervised_head(part_features)

                #for i in range(self.num_tiles):
                #    img = [image[i]]
                #    img, _ = self.tile_transform(img)
                #    part_features = self.backbone.body(img.tensors)                            
                #    part_features = self.jig_avg_pooling(part_features["3"])
                #    part_features = self.jig_fc_layers(torch.flatten(part_features, start_dim=1))
                #    feature_stack.append(part_features)
                #jig_features = torch.stack(feature_stack) 
                #jig_pred = self.self_supervised_head(jig_features)
                #jig_pred_list.append(jig_pred)
        
            else:

                for image in images:

                    ext_inputs = self._tile_processing(image, targets)
                    feature_stack = []
                    count = 0

                    for i, tile in enumerate(image):
                        if i in ext_inputs["tile_idx"]:

                            targets[0]["masks"] = ext_inputs["tile_masks"][count]
                            targets[0]["boxes"] = ext_inputs["tile_boxes"][count]
                            targets[0]["labels"] = ext_inputs["tile_labels"][count]
                            # glorified resize 
                            tile, targets = self.tile_transform([tile], targets)

                            part_features = self.backbone.body(tile.tensors)
                            features = self.backbone.fpn(part_features)

                            part_features = self.jig_avg_pooling(part_features["3"])
                            part_features = self.jig_fc_layers(torch.flatten(part_features, start_dim=1))
                            feature_stack.append(part_features)

                            if isinstance(features, torch.Tensor):
                                features = OrderedDict([("0", features)])

                            proposals, proposal_losses = self.rpn(tile, features, targets)

                            detections, detector_losses = self.roi_heads(features, proposals, tile.image_sizes, targets)

                            losses = {}
                            losses.update(detector_losses)
                            losses.update(proposal_losses)

                            for key in acc_losses.keys():
                                acc_losses[key] += losses.get(key, 0)
                            acc_loss_count += 1
                            count += 1
                        else:                            
                            tile, _ = self.tile_transform([tile])
                            part_features = self.backbone.body(tile.tensors)                            
                            part_features = self.jig_avg_pooling(part_features["3"])
                            part_features = self.jig_fc_layers(torch.flatten(part_features, start_dim=1))
                            feature_stack.append(part_features) 

                    jig_features = torch.stack(feature_stack) 
                    jig_features =  jig_features.permute(1,0,2)
                    jig_pred = self.self_supervised_head(jig_features)
                    jig_pred_list.append(jig_pred)

                jig_pred_tensor = torch.stack(jig_pred_list)
                jig_pred_tensor = jig_pred_tensor.squeeze(1)

            if targets is None:
                return jig_pred_tensor
            else:
                for key in acc_losses.keys():
                    acc_losses[key] /= acc_loss_count
                return acc_losses, jig_pred_tensor

        # code for validation and test
        else:
            original_image_sizes: List[Tuple[int, int]] = []
            for img in images:
                val = img.shape[-2:]
                torch._assert(
                    len(val) == 2,
                    f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
                )
                original_image_sizes.append((val[0], val[1]))

        # this is to carry out convetional mrcnn
        images, targets = self.transform(images, targets)

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)

        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if self.training:
            return losses
        return detections
            
    def _tile_processing(self, images, targets):
        """ Detials """

        ext_inputs = {
            "tile_idx": [],
            "tile_masks": [],
            "tile_boxes": [],
            "tile_labels": [],
        }

        device = images[0].device
        images = images.unsqueeze(0)


        for image, target in zip(images, targets):
            for tile_id in range(image.shape[0]):
                if target["labels"][:, tile_id].sum() > 0:

                    tile_masks = []
                    tile_boxes = []
                    tile_labels = []

                    masks_idx = (target["labels"][:, tile_id].nonzero(as_tuple=True)[0]).tolist()
                    for mask_idx in masks_idx:
                        tile_masks.append(target["masks"][tile_id, mask_idx])
                        tile_boxes.append(target["boxes"][mask_idx])
                        tile_labels.append(target["labels"][mask_idx, tile_id].item())

                    tile_masks = torch.stack(tile_masks)
                    tile_boxes = torch.stack(tile_boxes)
                    tile_labels = torch.tensor(tile_labels)

                    ext_inputs["tile_idx"].append(tile_id)
                    ext_inputs["tile_masks"].append(tile_masks)
                    ext_inputs["tile_boxes"].append(tile_boxes)
                    ext_inputs["tile_labels"].append(tile_labels.to(device))

        return ext_inputs

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
    tile_min = min_size // 3
    tile_max = max_size // 3

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
    
    #if drop_out:
    #    backbone.body.layer4.add_module("dropout", nn.Dropout(drop_out))

    model = JigMaskRCNN(backbone, num_classes, num_tiles, num_permutations, tile_max=tile_max, tile_minx=tile_min, max_size=max_size, min_size=min_size)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layers, num_classes)
    
    return model