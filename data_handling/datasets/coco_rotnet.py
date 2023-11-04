"""
Detials
"""
# imports
import json
import os

import torch
import torchvision.transforms as T
import torch.utils.data as data
import torch.nn.functional as F
from pycocotools.coco import COCO
from PIL import Image
import numpy as np

# class
class COCORotDataset(data.Dataset):
    """ Detials """
    def __init__(self, cfg, type, seed=42):
        self.cfg = cfg
        self.type = type
        self._extract_config()

        self._initialize_params()

    def _extract_config(self):
        """Detials"""
        self.sup_source = self.cfg["source"][0]
        self.type_dir = self.cfg["params"][self.type]["dir"]
        self.type_json = self.cfg["params"][self.type]["json"]
        self.num_rotations = self.cfg["params"]["num_rotations"]

    def _initialize_params(self):
        """Detials"""
        self.coco = COCO(os.path.join(self.sup_source, self.type_dir, self.type_json))
        self.dir = os.path.join(self.sup_source, self.type_dir)
        self.ids = list(self.coco.imgs.keys())
        self.rot_deg = np.linspace(0, 360, self.num_rotations + 1).tolist()[:-1]

    def __getitem__(self, idx):
        """Details"""
        # laoding image and related coco data
        img_id = self.ids[idx]
        file_name = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.dir, file_name)).convert('RGB')

        # generate rotation anlge theta
        theta = np.random.choice(self.rot_deg, size=1)[0]

        # rotate image
        image_tensor = self._to_tensor(img)
        image_tensor = self._rotate_tensor(image_tensor, theta)

        # get mrcnn targets
        mrcnn_target = self._coco_target_collection(img_id, theta, idx)

        # get rotnet target
        rot_target = torch.zeros(self.num_rotations)
        rot_target[self.rot_deg.index(theta)] = 1

        return image_tensor, mrcnn_target, rot_target

    def _coco_target_collection(self, img_id, theta, idx):
        """Detials"""
        # collecting ids and annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # initialise target lists and targ dict
        labels, boxes, masks_list, areas, iscrowds = [], [], [], [], []
        target = {}

        for ann in anns:
            # collecting target data
            labels.append(ann["category_id"])
            areas.append(ann["area"]) 
            boxes.append([ann["bbox"][0], ann["bbox"][1], ann["bbox"][0]+ann["bbox"][2], ann["bbox"][1]+ann["bbox"][3]])
            
            # formatting and collecting iscrowd id's
            if ann["iscrowd"]:
                iscrowds.append(1)
            else:
                iscrowds.append(0)

            # formatting mask to tensor and collecting masks
            mask = self.coco.annToMask(ann)
            mask == ann['category_id']
            masks_list.append(torch.from_numpy(mask)) 

        masks = torch.stack(masks_list, 0)  
        rotated_masks = self._rotate_tensor(masks, theta)  
        rotated_boxes = self._rotate_boxes(boxes, rotated_masks.shape, theta)
        rotated_boxes = torch.as_tensor(rotated_boxes, dtype=torch.float32)

        # building target_dict
        target["boxes"] = rotated_boxes
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["masks"] = rotated_masks
        target["image_id"] = torch.tensor([idx])
        target["area"] = torch.as_tensor(areas, dtype=torch.int64)
        target["iscrowd"] = torch.as_tensor(iscrowds, dtype=torch.int64)

        return(target)
    
    def _rotate_tensor(self, tensor, theta):
        """
        Detials
        """
        if theta == 0:
            return tensor
        
        # swap indicates weather height and width should be swapped when handling tensor rotation
        if theta in [90, 270]:
            swap = True
        else:
            swap = False

        # get tensor image data type
        dtype = tensor.dtype
        theta *= np.pi/180
        theta = torch.tensor(theta)

        if dtype == torch.uint8:
            tensor = tensor.float()
            dtype = tensor.dtype
            mask_flag = True
        else:
            mask_flag = False

        tensor = tensor.unsqueeze(0)
        rotation_matrix = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                                        [torch.sin(theta), torch.cos(theta), 0]])
        rotation_matrix = rotation_matrix[None, ...].type(dtype).repeat(tensor.shape[0], 1, 1)

        # this code handles the input the the affine_grid function for correctly carrying out the rotation
        H, W = tensor.shape[-2], tensor.shape[-1]
        if swap:
            H, W = W, H
        tensor_shape = [tensor.shape[0], tensor.shape[1], H, W]
        grid = F.affine_grid(rotation_matrix,
                                     tensor_shape,
                                     align_corners=True).type(dtype)
        rotated_tensor = F.grid_sample(tensor, grid, align_corners=True)
        rotated_tensor = rotated_tensor.squeeze(0)

        if mask_flag:
            rotated_tensor = (rotated_tensor > 0.5).byte()

        # returning rotated image tensor
        return rotated_tensor
    
    def _rotate_boxes(self, boxes, mask_shape, theta):
        """
        Detials
        """
        # return bounding boxes as if no rotation needs to be applied
        if theta == 0:
            return boxes
    
        H, W = mask_shape[-2], mask_shape[-1]
        swap_dims = theta in [90, 270]
        if swap_dims:
            W, H = H, W
        cx, cy = W / 2, H / 2  # Image center

        rotated_boxes = []
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            xmin -= cx
            xmax -= cx
            ymin -= cy
            ymax -= cy
        
            # Define rotation matrix for counter-clockwise rotation
            angle = np.deg2rad(theta)
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])

            # Rotate the corners of the bounding box
            corners = np.array([
                [xmin, ymin],
                [xmax, ymin],
                [xmax, ymax],
                [xmin, ymax]
            ])

            rotated_corners = corners.dot(rotation_matrix)

            # Find min/max to get the rotated bounding box
            xmin_new, ymin_new = rotated_corners.min(axis=0)
            xmax_new, ymax_new = rotated_corners.max(axis=0)

            if swap_dims:
                xmin_new += cy
                xmax_new += cy
                ymin_new += cx
                ymax_new += cx
            else:
                xmin_new += cx
                xmax_new += cx
                ymin_new += cy
                ymax_new += cy

            new_box = [xmin_new, ymin_new, xmax_new, ymax_new]
            rotated_boxes.append(new_box)
    
        #return rotated_boxes
        return(rotated_boxes)

    def _to_tensor(self, img):
        transform = T.ToTensor()
        ten_img = transform(img)
        return ten_img
    
    def __len__(self):
        return len(self.ids)
      
def COCO_collate_function(batch):
    """Detials"""
    return tuple(zip(*batch))