"""
Detials
"""
# imports
import torch
import torch.nn as nn
import torchvision

# model
class Multi_Mask_RCNN(nn.Module):
    """
    Detials
    """
    def __init__(self, cfg):
        super().__init__()

        if cfg["model"]["pre_trained"]:
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
            num_classes=1)

        self.fc_layers = nn.Sequential(
            nn.Linear(266240, 2048, bias=False),
            nn.ReLU(inplace=True),
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
        
    def forward(self, flag, x, y=None):

        if flag == "mask":
            x = self.Mask_RCNN(x)
            return(x)
        elif flag == "ss":
            x = self.Mask_RCNN.backbone.body(x)
            x = x["3"]
            x = torch.flatten(x, start_dim = 1)
            x = self.fc_layers(x)
            x = self.rot_classifier(x)

            return(x)

# test
if __name__ == "__main__":

    model = Multi_Mask_RCNN({"model":{"pre_trained": True}})
    print(model)
    #model.train()
    #x = [torch.rand(3, 400, 300), torch.rand(3, 400, 300)]
    #pred = model("mask", x)
    #print(pred)
