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


class Mask_RCNN_Resnet_50_FPN(nn.Module):
    """
    Detials
    """
    def __init__(self, cfg):
        super().__init__()

        self.num_classes = cfg["num_classes"]

        # backbone selecting
        if cfg["pre_trained"]:
            self.backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
                         backbone_name="resnet50",
                         weights="ResNet50_Weights.DEFAULT",
                         trainable_layers=cfg["trainable_layers"])
        else:
            self.backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
                backbone_name="resnet50",
                weights=False,
                trainable_layers=cfg["trainable_layers"])

        #anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
        #                                   aspect_ratios=((0.5, 1.0, 2.0),))
        ## define roi features for roi cropping 
        #roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
        #                                                output_size=7,
        #                                                sampling_ratio=2)
        ## define mask pooler for mask 
        #mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
        #                                                     output_size=14,
        #                                                     sampling_ratio=2)               

        self.Mask_RCNN = torchvision.models.detection.MaskRCNN(
            self.backbone,
            num_classes=self.num_classes)#,
            #rpn_anchor_generator=anchor_generator,
            #box_roi_pool=roi_pooler,
            #mask_roi_pool=mask_roi_pooler)

        # get number of input features for the classifier
        in_features = self.Mask_RCNN.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.Mask_RCNN.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        # now get the number of input features for the mask classifier
        in_features_mask = self.Mask_RCNN.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        self.Mask_RCNN.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                           hidden_layer,
                                                           self.num_classes)
        
        if cfg["loaded"]:    
            checkpoint = torch.load(cfg["load_source"])
            self.Mask_RCNN.backbone.load_state_dict(checkpoint["state_dict"], strict=False)
            print("LOADED SSL")
    
    def forward(self, x, y=None):

        if y == None:
            x = self.Mask_RCNN(x)
            return x
        else:
            x = self.Mask_RCNN(x, y)
            return x

# test
if __name__ == "__main__":

    model = Mask_RCNN_Resnet_50_FPN({"model":{"pre_trained": True}})
    model.eval()
    x = [torch.rand(3, 400, 300), torch.rand(3, 400, 300)]
    pred = model(x)
