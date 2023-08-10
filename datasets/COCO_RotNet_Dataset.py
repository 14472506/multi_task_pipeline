"""
Title:

Function:

Edited by:
"""
# imports
# standard library
import json
import os

# third party
import torch
import torchvision.transforms as T
import torch.utils.data as data
import torch.nn.functional as F
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from PIL import Image
import numpy as np

# local packages
from utils import basic_square_crop, resize

# class
class COCORotDataset(data.Dataset):
    """
    Detials
    """
    def __init__(self, cfg, type, seed=42):
        self.cfg = cfg
        self.seed = seed
        self.type = type

        self._extract_config()

        self._initialize_params()

    def _extract_config(self):
        """Detials"""
        self.source = self.cfg.get("source_joint", "")
        self.params = self.cfg.get("params", "")

    def _initialize_params(self):
        """Detials"""
        self.coco = COCO(os.path.join(
                self.source,
                self.params.get(self.type, "").get("dir", ""),
                self.params.get(self.type, "").get("json", "")
            ))
        self.dir = os.path.join(self.source, self.params.get(self.type, "").get("dir", ""))
        self.ids = list(self.coco.imgs.keys())
        self.num_rotations = self.params.get("num_rotations", 0)
        self.rot_deg = np.linspace(0, 360, self.num_rotations + 1).tolist()[:-1]

    def __getitem__(self, idx):
        """Details"""
        # laoding image
        img_id = self.ids[idx]
        file_name = self.coco.loadImgs(img_id)[0]['file_name']
        
        mrcnn_img = Image.open(os.path.join(self.dir, file_name)).convert('RGB')
        mrcnn_target = self._coco_target_collection(img_id, idx)

        rot_tensor, rot_target = self._generate_rotnet_data(mrcnn_img)

        mrcnn_tensor = self._to_tensor(mrcnn_img)

        return mrcnn_tensor, mrcnn_target, rot_tensor, rot_target

    def _coco_target_collection(self, img_id, idx):
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

        # building target_dict
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["masks"] = torch.stack(masks_list, 0)
        target["image_id"] = torch.tensor([idx])
        target["area"] = torch.as_tensor(areas, dtype=torch.int64)
        target["iscrowd"] = torch.as_tensor(iscrowds, dtype=torch.int64)

        return(target)

    def _generate_rotnet_data(self, img):
        """Detials"""
        # shape and resize image
        rot_img = basic_square_crop(img)
        rot_img = resize(rot_img)

        # conver image to tensore
        tensor = self._to_tensor(rot_img).unsqueeze(0)
        dtype = tensor.dtype

        # generate rotation angle
        theta_idx = np.random.choice(self.rot_deg, size=1)[0]
        theta = theta_idx*(np.pi/180)
        theta = torch.tensor(theta)

        # get z axis rotation matrix
        rotation_matrix = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                                        [torch.sin(theta), torch.cos(theta), 0]])
        rotation_matrix = rotation_matrix[None, ...].type(dtype).repeat(tensor.shape[0], 1, 1)
        
        # appling rotation
        grid = F.affine_grid(rotation_matrix,
                                     tensor.shape,
                                     align_corners=True).type(dtype)
        rotated_torch_tensor = F.grid_sample(tensor, grid, align_corners=True)

        target = torch.zeros(self.num_rotations)
        target[self.rot_deg.index(theta_idx)] = 1

        return rotated_torch_tensor, target

    def _to_tensor(self, img):
        transform = T.ToTensor()
        ten_img = transform(img)
        return ten_img
    
    def __len__(self):
        return len(self.ids)
  
def COCO_collate_function(batch):
    """Detials"""
    return tuple(zip(*batch)) 