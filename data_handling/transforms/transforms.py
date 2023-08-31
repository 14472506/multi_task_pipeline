"""
Detials
"""
# imports
import albumentations as A 
from PIL import Image

# class
class Transforms():
    """ Detials """
    def __init__(self, cfg):
        """ Detials """ 
        self.cfg = cfg
        self.model = self.cfg["params"]["model_name"]

    def transforms(self):
        """ Detials """
        transform_selector = {
            "rotnet_resnet_50": self._rotnet_transforms
        }
        return transform_selector[self.model]()
    
    def _rotnet_transforms(self):
        """ Detials """
        transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ToGray(p=0.2),
            A.Downscale(p=0.1),
            A.ColorJitter(p=0.2),
            A.RandomBrightnessContrast(p=0.2)
            #A.RandomResizedCrop(300, 300, p=0.1)
        ], p=1)
        return transforms

