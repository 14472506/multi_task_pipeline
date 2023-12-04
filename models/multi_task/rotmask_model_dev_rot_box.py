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
from torch import nn, Tensor
import torchvision
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead, _default_anchorgen
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, MaskRCNNHeads
from torchvision.models.detection.rpn import concat_box_prediction_layers, RPNHead
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.roi_heads import fastrcnn_loss, maskrcnn_loss, maskrcnn_inference

# local packages

# classes
class NoLossRPN(torchvision.models.detection.rpn.RegionProposalNetwork):
    """ Detials """
    def __init__(self,
        anchor_generator,
        head,
        # Faster-RCNN Training
        fg_iou_thresh,
        bg_iou_thresh,
        batch_size_per_image,
        positive_fraction,
        # Faster-RCNN Inference
        pre_nms_top_n,
        post_nms_top_n,
        nms_thresh,
        score_thresh,
        ) -> None:
        # No modifications to be made in the class init
        super(NoLossRPN, self).__init__(        
            anchor_generator,
            head,
            # Faster-RCNN Training
            fg_iou_thresh,
            bg_iou_thresh,
            batch_size_per_image,
            positive_fraction,
            # Faster-RCNN Inference
            pre_nms_top_n,
            post_nms_top_n,
            nms_thresh,
            score_thresh,
        )

    def forward(
        self,
        images: ImageList,
        features: Dict[str, Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None,
        ) -> Tuple[List[Tensor], Dict[str, Tensor]]:

        """
        Args:
            images (ImageList): images for which we want to compute the predictions
            features (Dict[str, Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (List[Dict[str, Tensor]]): ground-truth boxes present in the image (optional).
                If provided, each element in the dict should contain a field `boxes`,
                with the locations of the ground-truth boxes.

        Returns:
            boxes (List[Tensor]): the predicted boxes from the RPN, one Tensor per
                image.
            losses (Dict[str, Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        # RPN uses all feature maps that are available
        features = list(features.values())
        objectness, pred_bbox_deltas = self.head(features)
        anchors = self.anchor_generator(images, features)

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        losses = {}
        if self.training:
            if targets is None:
                loss_objectness = None
                loss_rpn_box_reg = None
            else:
                labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
                regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
                loss_objectness, loss_rpn_box_reg = self.compute_loss(
                    objectness, pred_bbox_deltas, labels, regression_targets
            )
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }
        return boxes, losses

class NoLossRoIHeads(torchvision.models.detection.roi_heads.RoIHeads):
    """ Details """
    def __init__(self, rot_predictor=None, **kwargs):
        super().__init__(**kwargs)

        self.rot_predictor = rot_predictor
    
    def forward(
        self,
        features,  # type: Dict[str, Tensor]
        proposals,  # type: List[Tensor]
        image_shapes,  # type: List[Tuple[int, int]]
        targets=None,  # type: Optional[List[Dict[str, Tensor]]]
    ):
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                if not t["boxes"].dtype in floating_point_types:
                    raise TypeError(f"target boxes must of float type, instead got {t['boxes'].dtype}")
                if not t["labels"].dtype == torch.int64:
                    raise TypeError(f"target labels must of int64 type, instead got {t['labels'].dtype}")
                if self.has_keypoint():
                    if not t["keypoints"].dtype == torch.float32:
                        raise TypeError(f"target keypoints must of float type, instead got {t['keypoints'].dtype}")

        if self.training and targets is not None:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            matched_idxs, labels, regression_targets = None, None, None
            # Do RotNet Here and Return Results

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)
        rotation_logits = self.rot_predictor(box_features)

        result: List[Dict[str, torch.Tensor]] = []
        losses = {}
        if self.training:
            if labels is None:
                losses = {"loss_classifier": None, "loss_box_reg": None, "rotation_logits": rotation_logits}
            else:
                if regression_targets is None:
                    raise ValueError("regression_targets cannot be None")
                loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
                losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg, "rotation_logits": rotation_logits}
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        if self.has_mask():
            if self.training and targets is not None:
                mask_proposals = [p["boxes"] for p in result]
                if self.training:
                    if matched_idxs is None:
                        raise ValueError("if in training, matched_idxs should not be None")

                    # during training, only focus on positive boxes
                    num_images = len(proposals)
                    mask_proposals = []
                    pos_matched_idxs = []
                    for img_id in range(num_images):
                        pos = torch.where(labels[img_id] > 0)[0]
                        mask_proposals.append(proposals[img_id][pos])
                        pos_matched_idxs.append(matched_idxs[img_id][pos])
                else:
                    pos_matched_idxs = None

                if self.mask_roi_pool is not None:
                    mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
                    mask_features = self.mask_head(mask_features)
                    mask_logits = self.mask_predictor(mask_features)
                else:
                    raise Exception("Expected mask_roi_pool to be not None")

                loss_mask = {}
                if self.training:
                    if targets is None or pos_matched_idxs is None or mask_logits is None:
                        raise ValueError("targets, pos_matched_idxs, mask_logits cannot be None when training")

                    gt_masks = [t["masks"] for t in targets]
                    gt_labels = [t["labels"] for t in targets]
                    rcnn_loss_mask = maskrcnn_loss(mask_logits, mask_proposals, gt_masks, gt_labels, pos_matched_idxs)
                    loss_mask = {"loss_mask": rcnn_loss_mask}
                else:
                    labels = [r["labels"] for r in result]
                    masks_probs = maskrcnn_inference(mask_logits, labels)
                    for mask_prob, r in zip(masks_probs, result):
                        r["masks"] = mask_prob

                losses.update(loss_mask)
            else:
                losses.update({"loss_mask": None})
        return result, losses

class RotNetHead(torch.nn.Module):
    """
    RotNetHead to be added to MaskRCNN class. the feature extractor is alread in the MaskRCNN class
    Which the RotMaskRCNN class will inherits from. The rotnet head in this class takes the
    features provided from the RotMaskRCNN model and returns the classificiation.  
    """
    def __init__(self, 
                 in_channels,
                 num_rots=4,
                 #batch_norm=True,
                 #drop_out=0.5
                 ):
        super().__init__()
        self.rot_score = nn.Linear(in_channels, num_rots)

        #self.classifier = nn.Sequential(
        #    nn.Dropout() if drop_out > 0. else nn.Identity(),
        #    nn.Linear(1000, 4096, bias=False if batch_norm else True),
        #    nn.BatchNorm1d(4096) if batch_norm else nn.Identity(),
        #    nn.ReLU(inplace=True),
        #    nn.Dropout() if drop_out > 0. else nn.Identity(),
        #    nn.Linear(4096, 4096, bias=False if batch_norm else True),
        #    nn.BatchNorm1d(4096) if batch_norm else nn.Identity(),
        #    nn.ReLU(inplace=True),
        #    nn.Dropout() if drop_out > 0. else nn.Identity(),
        #    nn.Linear(4096, 1000, bias=False if batch_norm else True),
        #    nn.BatchNorm1d(1000) if batch_norm else nn.Identity(),
        #    nn.ReLU(inplace=True),
        #    nn.Linear(1000, num_rots))
        #self.classifier = nn.Sequential(*[child for child in self.classifier.children() if not isinstance(child, nn.Identity)])    
    
    def forward(self, x):
        x = x.flatten(start_dim=1)
        score = self.rot_score(x)

        return score

        #x = self.avg_pooling(x)
        #x = torch.flatten(x, start_dim=1)
        #x = self.fc_layers(x)
        #x = self.classifier(x)
        #return x
    
    @torch.jit.unused
    def eager_outputs(self, losses, detections, rot_pred):
        if self.training:
            return losses, rot_pred

        return detections

class RotMaskRCNN(torchvision.models.detection.MaskRCNN):
    """ Detials """
    def __init__(
        self,
        backbone,
        num_classes=91,
        # transform parameters
        min_size=800,
        max_size=1333,
        image_mean=None,
        image_std=None,
        # RPN parameters
        rpn_anchor_generator=None,
        rpn_head=None,
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
        rpn_score_thresh=0.0,
        # Box parameters
        box_roi_pool=None,
        box_head=None,
        box_predictor=None,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512,
        box_positive_fraction=0.25,
        bbox_reg_weights=None,
        # Mask parameters
        mask_roi_pool=None,
        mask_head=None,
        mask_predictor=None,
        # Adding Dual Mask and Box Predictors
        num_rotations=4,
        rot_predictor=None,
        **kwargs,      
        ):

        # >>> potential further definitions here
        super(RotMaskRCNN, self).__init__(
            backbone,
            num_classes,
            # transform parameters
            min_size,
            max_size,
            image_mean,
            image_std,
            # RPN parameters
            rpn_anchor_generator,
            rpn_head,
            rpn_pre_nms_top_n_train,
            rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_train,
            rpn_post_nms_top_n_test,
            rpn_nms_thresh,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_score_thresh,
            # Box parameters
            box_roi_pool,
            box_head,
            box_predictor,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            # Mask parameters
            mask_roi_pool,
            mask_head,
            mask_predictor,
            # Adding Dual Mask and Box Predictors
            num_rotations=4,
            rot_predictor=None,
            **kwargs, 
            ) 
        
        # For Faster R-CNN init
        out_channels = backbone.out_channels

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(out_channels * resolution**2, representation_size)

        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(representation_size, num_classes)
        
        # For Mask R-CNN
        if mask_roi_pool is None:
            mask_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2)

        if mask_head is None:
            mask_layers = (256, 256, 256, 256)
            mask_dilation = 1
            mask_head = MaskRCNNHeads(out_channels, mask_layers, mask_dilation)

        if mask_predictor is None:
            mask_predictor_in_channels = 256  # == mask_layers[-1]
            mask_dim_reduced = 256
            mask_predictor = MaskRCNNPredictor(mask_predictor_in_channels, mask_dim_reduced, num_classes)

        # For RotNet Head
        if rot_predictor is None:
            representation_size = 1024
            rot_predictor = RotNetHead(representation_size, num_rotations)

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        self.rpn = NoLossRPN(
            anchor_generator = rpn_anchor_generator,
            head = rpn_head,
            # Faster-RCNN Training
            fg_iou_thresh = rpn_fg_iou_thresh,
            bg_iou_thresh = rpn_bg_iou_thresh,
            batch_size_per_image = rpn_batch_size_per_image,
            positive_fraction = rpn_positive_fraction,
            # Faster-RCNN Inference
            pre_nms_top_n = rpn_pre_nms_top_n,
            post_nms_top_n = rpn_post_nms_top_n,
            nms_thresh = rpn_nms_thresh,
            score_thresh = rpn_score_thresh,
        )

        self.roi_heads = NoLossRoIHeads(
            # Box
            box_roi_pool = box_roi_pool,
            box_head = box_head,
            box_predictor = box_predictor,
            fg_iou_thresh = box_fg_iou_thresh,
            bg_iou_thresh = box_bg_iou_thresh,
            batch_size_per_image = box_batch_size_per_image,
            positive_fraction = box_positive_fraction,
            bbox_reg_weights = bbox_reg_weights,
            score_thresh = box_score_thresh,
            nms_thresh = box_nms_thresh,
            detections_per_img = box_detections_per_img,
            mask_roi_pool = mask_roi_pool,
            mask_head = mask_head,
            mask_predictor = mask_predictor,
            rot_predictor=rot_predictor
        )

        #self.roi_heads.rot_predictor = rot_predictor


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
                self.losses_flag = False            
            else:
                self.losses_flag = True
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

        features = self.backbone(images.tensors)

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
            if self.losses_flag:
                return losses
            else:
                return losses
        else:
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

    rpn_anchor_generator = _default_anchorgen()
    rpn_head = RPNHead(backbone.out_channels, rpn_anchor_generator.num_anchors_per_location()[0], conv_depth=2) 
    
    model = RotMaskRCNN(backbone, 
                        num_classes=num_classes, 
                        rpn_anchor_generator=rpn_anchor_generator, 
                        rpn_head=rpn_head, num_rotations=num_rots, 
                        max_size=max_size, 
                        min_size=min_size)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layers, num_classes)
    
    return model