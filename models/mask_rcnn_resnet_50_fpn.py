"""
Mask R-CNN baseline Model
"""
# imports
import torch
import torch.nn as nn
import torchvision

class Mask_RCNN_Resnet_50_FPN(nn.Module):
    """
    Detials
    """
    def __init__(self, cfg):
        super().__init__()

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

        self.Mask_RCNN = torchvision.models.detection.MaskRCNN(
            self.backbone,
            num_classes=cfg["num_classes"])
    
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
