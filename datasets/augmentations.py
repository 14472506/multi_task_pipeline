"""
Detials
"""
# libraries
import albumentations as A 
from PIL import Image

# class
class Augmentations():
    """
    The class contains the augmentations for different datasets
    """
    def __init__(self, loader_type, aug_type=None):
        """
        initialises atributes that indentify the loader type that the augmentations are requred for
        and the specific set of augmentations to be applied if any augmentation variants are 
        available. 
        """
        self.loader_type = loader_type
        self.aug_type = aug_type

    def aug_loader(self):
        """
        The augmentation loader method is called to select and return the specific set of augmentations 
        """
        if self.loader_type == "Mask_RCNN":
            transforms = self.Mask_RCNN_1()
            return transforms
        if self.loader_type == "RotNet":
            transforms = self.RotNet_1()
            return transforms
        if self.loader_type == "Jigsaw":
            transforms = self.Jigsaw_1()
            return transforms
        if self.loader_type == "multi_task":
            transforms = self.Mask_RCNN_Rot()
            return transforms

    def Mask_RCNN_1(self):
        """
        Implementation of Mask R-CNN augmentations
        """
        transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.OneOf([A.RandomBrightnessContrast(p=0.2),                     
                     A.ToGray(p=0.3)                     
                     ], p=1)
        ], p=1, additional_targets={'image0': 'image', 'mask0': 'mask'})
        
        return transforms
    
    def RotNet_1(self):
        """
        Implmentation of RotNet augmentations
        """
        transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ToGray(p=0.2),
            A.Downscale(p=0.1),
            A.ColorJitter(p=0.2),
            A.RandomBrightnessContrast(p=0.2),
            A.RandomResizedCrop(300, 300, p=0.1)
        ], p=1)

        return transforms

    def Jigsaw_1(self):
        pass
        """
        Implementation of Jigsaw augmentations 
        """
        transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ToGray(p=0.2),
            A.Downscale(p=0.1),
            A.ColorJitter(p=0.2),
            A.RandomBrightnessContrast(0.2)
        ], p=1)

        return transforms
    
    def Mask_RCNN_Rot(self):
        """
        Implementation of Mask R-CNN augmentations
        """
        transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.OneOf([A.RandomBrightnessContrast(p=0.2),                     
                     A.ToGray(p=0.3)                     
                     ], p=1)
        ], p=1, additional_targets={'image0': 'mrcnn_image', 'mask0': 'mask', "image1": "rot_img"})