"""
Detials
"""
# imports
import torch
import torch.nn as nn
import torchvision

# model
class Multi_task_RotNet_Mask_RCNN_Resnet_50_FPN(nn.Module):
    """
    Detials
    """
    def __init__(self, cfg):
        super().__init__()

        if cfg["backbone_type"] == "pre-trained":
            self.backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
                backbone_name="resnet50",
                weights="ResNet50_Weights.DEFAULT",
                trainable_layers=5)
        else:
            self.backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
                backbone_name="resnet50",
                weights=False,
                trainable_layers=5)

        self.Mask_RCNN = torchvision.models.detection.MaskRCNN(
            self.backbone,
            num_classes=2)

        self.avg_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(2048, 1000, bias=False)
        )

        self.rot_classifier = nn.Sequential(
                                nn.Dropout() if 0.5 > 0. else nn.Identity(),
                                nn.Linear(1000, 4096, bias=False),
                                #nn.BatchNorm1d(4096) if batch_norm else nn.Identity(),
                                nn.ReLU(inplace=True),
                                nn.Dropout() if 0.5 > 0. else nn.Identity(),
                                nn.Linear(4096, 4096, bias=False),
                                #nn.BatchNorm1d(4096) if batch_norm else nn.Identity(),
                                nn.ReLU(inplace=True),
                                nn.Dropout() if 0.5 > 0. else nn.Identity(),
                                nn.Linear(4096, 1000, bias=False),
                                #nn.BatchNorm1d(1000) if batch_norm else nn.Identity(),
                                nn.ReLU(inplace=True),
                                nn.Linear(1000, 4))
        
    def forward(self, flag, x, y=None, xi=None):

        if flag == "mask":
            x1 = self.Mask_RCNN(x, y)

            x2 = self.Mask_RCNN.backbone.body(xi)
            x2 = x2["3"]
            x2 = self.avg_pooling(x2)
            x2 = torch.flatten(x2, start_dim = 1)
            x2 = self.fc_layers(x2)
            x2 = self.rot_classifier(x2)            
            
            return(x1, x2)

        elif flag == "ss":
            x = self.Mask_RCNN.backbone.body(x)
            x = x["3"]
            x = self.avg_pooling(x)
            x = torch.flatten(x, start_dim = 1)
            x = self.fc_layers(x)
            x = self.rot_classifier(x)

            return(x)

# test
if __name__ == "__main__":

    model = Multi_task_RotNet_Mask_RCNN_Resnet_50_FPN({"backbone_type": "pre-trained"})
    model.train()
    x = [torch.rand(3, 400, 300), torch.rand(3, 400, 400)]
    pred = model("ss", torch.rand(3, 1000, 1000))
    print(pred)
