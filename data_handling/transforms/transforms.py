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
        self.model = self.cfg["model_name"]

    def transforms(self):
        """ Detials """
        transform_selector = {
            "rotnet_resnet_50": self._rotnet_transforms,
            "mask_rcnn": self._maskrcnn_tranforms,
            "rotmask_multi_task": self._multitask_transforms
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
    
    def _maskrcnn_tranforms(self):
        """ Detials """
        transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(p=0.3),
                A.ToGray(p=0.3)
            ], p=1)
            #A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=25, p=0.3)
        ], p=1,
        additional_targets={'image0': 'image'})
        return transforms

    def _multitask_transforms(self):
        """ Detials """
        transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(p=0.3),
                A.ToGray(p=0.3)
            ], p=1)
            #A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=25, p=0.3)
        ], p=1,
        additional_targets={'image0': 'image'})
        return transforms
