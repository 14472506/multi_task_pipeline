"""
Detials 
"""
# imports
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
import torchvision.transforms as T
import numpy as np
import PIL

# class
class RotNetWrapper(torch.utils.data.Dataset):
    """ Detials """
    def __init__(self, dataset, transforms):
        """ Detials """
        self.dataset = dataset
        self.transforms = transforms

    def __getitem__(self, idx):
        """ Detials """
        image, label = self.dataset[idx]

        pil_trans = T.ToPILImage()
        pil = pil_trans(image)
        np_img = np.array(pil)
        aug_data = self.transforms(image=np_img)['image']
        transformed = torch.from_numpy(aug_data)
        transformed = transformed.permute(2,0,1)
        transformed = transformed.to(dtype=torch.float32) / 255.0

        return transformed, label

    def __len__(self):
        """ Details """
        return len(self.dataset)
    
def wrappers(model_type):
    """ Detials """
    transform_select = {
        "rotnet_resnet_50": RotNetWrapper
    }
    return transform_select[model_type]