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
    
class InstanceWrapper(torch.utils.data.Dataset):
    """ detials """
    def __init__(self, dataset, transforms):
        self.dataset = dataset
        self.transforms = transforms

    def __getitem__(self, idx):
        """Details"""
        # Getting image
        mrcnn_tensor, mrcnn_target = self.dataset[idx]

        # converting tensors to arrays
        to_img = T.ToPILImage()        
        mrcnn_img = to_img(mrcnn_tensor)
        mrcnn_arr = np.array(mrcnn_img)

        np_masks = []
        for mask, box in zip(mrcnn_target["masks"], mrcnn_target["boxes"]): 
            mask_img = to_img(mask)
            # append values to accumulated lists
            np_masks.append(np.array(mask_img))

        # applying augmentations
        aug_data = self.transforms(image=mrcnn_arr, masks=np_masks)

        boxes_list = []
        for mask in aug_data["masks"]:
            box = self._mask_to_bbox(mask)
            if box == None:
                pass
            else:
                boxes_list.append(box)

        # extracting auged data
        mrcnn_transformed = torch.from_numpy(aug_data["image"])
        mrcnn_transformed = mrcnn_transformed.permute(2,0,1)
        mrcnn_transformed = mrcnn_transformed.to(dtype=torch.float32) / 255.0
        mrcnn_target["masks"] = torch.stack([torch.tensor(arr) for arr in aug_data["masks"]])
        mrcnn_target["boxes"] = torch.as_tensor(boxes_list, dtype=torch.float32)
        
        return mrcnn_transformed, mrcnn_target
    
    def _mask_to_bbox(self, binary_mask):
        """ Details """
        # Get the axis indices where mask is active (i.e., equals 1)
        rows, cols = np.where(binary_mask == 1)

        # If no active pixels found, return None
        if len(rows) == 0 or len(cols) == 0:
            return None

        # Determine the bounding box coordinates
        x_min = np.min(cols)
        y_min = np.min(rows)
        x_max = np.max(cols)
        y_max = np.max(rows)

        return [x_min, y_min, x_max, y_max]

    def __len__(self):
        """ Details """
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

        # converting tensors to arrays
        to_img = T.ToPILImage()        
        mrcnn_img = to_img(mrcnn_tensor)
        rot_img = to_img(rot_tensor)

        mrcnn_arr = np.array(mrcnn_img)
        rot_arr = np.array(rot_img)

        np_masks = []
        for mask, box in zip(mrcnn_target["masks"], mrcnn_target["boxes"]): 
            mask_img = to_img(mask)
            # append values to accumulated lists
            np_masks.append(np.array(mask_img))

        # applying augmentations
        aug_data = self.transforms(image=mrcnn_arr, image0=rot_arr, masks=np_masks)

        boxes_list = []
        for mask in aug_data["masks"]:
            box = self._mask_to_bbox(mask)
            if box == None:
                pass
            else:
                boxes_list.append(box)

        # extracting auged data
        mrcnn_transformed = torch.from_numpy(aug_data["image"])
        mrcnn_transformed = mrcnn_transformed.permute(2,0,1)
        mrcnn_transformed = mrcnn_transformed.to(dtype=torch.float32) / 255.0
        rot_transformed = torch.from_numpy(aug_data["image0"])
        rot_transformed = rot_transformed.permute(2,0,1)
        rot_transformed = rot_transformed.to(dtype=torch.float32) / 255.0
        mrcnn_target["masks"] = torch.stack([torch.tensor(arr) for arr in aug_data["masks"]])
        mrcnn_target["boxes"] = torch.as_tensor(boxes_list, dtype=torch.float32)
        
        return mrcnn_transformed, mrcnn_target, rot_transformed, rot_target
    
    def _mask_to_bbox(self, binary_mask):
        """ Details """
        # Get the axis indices where mask is active (i.e., equals 1)
        rows, cols = np.where(binary_mask == 1)

        # If no active pixels found, return None
        if len(rows) == 0 or len(cols) == 0:
            return None

        # Determine the bounding box coordinates
        x_min = np.min(cols)
        y_min = np.min(rows)
        x_max = np.max(cols)
        y_max = np.max(rows)

        return [x_min, y_min, x_max, y_max]

    def __len__(self):
        """ Details """
        return len(self.dataset)
    
def wrappers(model_type):
    """ Detials """
    transform_select = {
        "rotnet_resnet_50": RotNetWrapper,
        "mask_rcnn": InstanceWrapper,
        "rotmask_multi_task": MultiTaskWrapper
    }
    return transform_select[model_type]
