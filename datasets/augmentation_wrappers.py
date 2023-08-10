"""
Detials - contains wrappers for applying augmentations to split datasets in training
"""
# library imports
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
import torchvision.transforms as T
import numpy as np
import PIL

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
        rot_img = to_img(rot_tensor.squeeze(0))
         
        mrcnn_arr = np.array(mrcnn_img)
        rot_arr = np.array(rot_img)

        np_masks = []

        for mask in mrcnn_target["masks"]: 
            mask_img = to_img(mask)
            mask_array = np.array(mask_img)
            np_masks.append(mask_array)
        
        all_data = [mrcnn_arr] + [rot_arr] + np_masks

        augmented_data = []
        for i in all_data:
            transformed = self.transforms(image=i)['image']
            augmented_data.append(transformed)

        auged_mrcnn = augmented_data[0]
        auged_rot = augmented_data[1]

        masks = augmented_data[2:]
        boxes = []
        masks_list = []
        for mask in masks:
            print(mask)
            np.savetxt("mask.csv", mask, delimiter=",")
            boxes.append(self._mask_to_box(mask))
            masks_list.append(torch.from_numpy(mask))
        
        mrcnn_target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        mrcnn_target["masks"] = torch.stack(masks_list, 0)

        mrcnn_transformed = torch.from_numpy(auged_mrcnn)
        rot_transformed = torch.from_numpy(auged_rot)

        print("APPLIED")
        
        return mrcnn_transformed, mrcnn_target, rot_transformed, rot_target
    
    def _mask_to_box(self, binary_mask):
        """Detials"""
        # Find the rows and columns that contain True values
        rows = np.any(binary_mask, axis=1)
        cols = np.any(binary_mask, axis=0)
    
        # Find the minimum and maximum row and column indices
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        # Compute width and height
        bbox_width = x_max - x_min + 1
        bbox_height = y_max - y_min + 1

        return [x_min, y_min, bbox_width, bbox_height]
        
    def __len__(self):
        """Details"""
        return len(self.dataset)

    