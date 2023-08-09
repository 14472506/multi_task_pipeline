"""
Detials - contains wrappers for applying augmentations to split datasets in training
"""
# library imports
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
import torchvision.transforms as T
import numpy as np

# classes
class RotNetWrapper(torch.utils.data.Dataset):
    """
    Detials
    """
    def __init__(self, dataset, transforms):
        """
        Detials
        """
        self.dataset = dataset
        self.transforms = transforms

    def __getitem__(self, idx):
        """
        Detials
        """
        image, label = self.dataset[idx]

        pil_trans = T.ToPILImage()
        pil = pil_trans(image)
        np_img = np.array(pil)
        transformed = self.transforms(image=np_img)['image']
     
        return(transformed, label)

    def __len__(self):
        """
        Details
        """
        return len(self.dataset)
    
class JigsawWrapper(torch.utils.data.Dataset):
    """
    Detials
    """
    def __init__(self, dataset, transforms):
        """
        Detials
        """
        self.dataset = dataset
        self.transforms = transforms

    def __getitem__(self, idx):
        """
        Detials
        """
        # getting image
        image, label = self.dataset[idx]
        
        # prepare augmented image stack
        aug_stack = []
        
        # loop through base stack
        for i in image:
            pil_trans = T.ToPILImage()
            pil = pil_trans(i)
            np_img = np.array(pil)
            transformed = self.transforms(image=np_img)["image"]
            aug_stack.append(transformed)

        stack = torch.stack(aug_stack)
        image = stack

        return(image, label)

    def __len__(self):
        """
        Details
        """
        return len(self.dataset)    

class JigRotWrapper(torch.utils.data.Dataset):
    """
    Detials
    """
    def __init__(self, dataset, transforms):
        """
        Detials
        """
        self.dataset = dataset
        self.transforms = transforms

    def __getitem__(self, idx):
        """
        Detials
        """
        # getting image
        image, label1, label2 = self.dataset[idx]
        
        # prepare augmented image stack
        aug_stack = []
        
        # loop through base stack
        for i in image:
            pil_trans = T.ToPILImage()
            pil = pil_trans(i)
            np_img = np.array(pil)
            transformed = self.transforms(image=np_img)["image"]
            aug_stack.append(transformed)

        stack = torch.stack(aug_stack)
        image = stack

        return(image, label1, label2)

    def __len__(self):
        """
        Details
        """
        return len(self.dataset)
    
class MultiTaskWrapper(torch.utils.data.Dataset):
    """Details"""
    def __init__(self, dataset, transforms):
        self.dataset = dataset
        self.transforms = transforms

    def __getitem__(self, idx):
        """Details"""

        # Getting image
        mrcnn_tensor, mrcnn_target, rot_tensor, rot_target = self.dataset[idx]
        
        to_img = T.ToPILImage()        
        mrcnn_img = to_img(mrcnn_tensor)
        rot_img = to_img(rot_tensor)
            
        mrcnn_arr = np.array(mrcnn_img)
        rot_arr = np.array(rot_img)
        
        # Assuming mrcnn_target[0] is a tensor of masks
        masks_arr = mrcnn_target[0].numpy()
            
        transformed = self.transforms(image=mrcnn_arr, mask=masks_arr, image0=rot_arr)
        
        # Getting transformed images/masks
        mrcnn_arr_transformed = transformed['image']
        masks_arr_transformed = transformed['mask']
        rot_arr_transformed = transformed['image0']
        
        # Convert back to tensors
        mrcnn_tensor_transformed = torch.tensor(mrcnn_arr_transformed).permute(2, 0, 1)
        rot_tensor_transformed = torch.tensor(rot_arr_transformed).permute(2, 0, 1)
        masks_tensor_transformed = torch.tensor(masks_arr_transformed).permute(2, 0, 1)  # Assuming it's a 3D array
        
        # Returning the transformed tensors (and assuming mrcnn_target and rot_target structures remain the same)
        return mrcnn_tensor_transformed, (masks_tensor_transformed, *mrcnn_target[1:]), rot_tensor_transformed, rot_target
        
    def __len__(self):
        """Details"""
        return len(self.dataset)

    