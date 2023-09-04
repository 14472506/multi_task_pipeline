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
        # laoding image
        img_id = self.ids[idx]
        file_name = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.dir, file_name)).convert('RGB')

        mrcnn_target = self._coco_target_collection(img_id, idx)
        mrcnn_tensor = self._to_tensor(img)
        rot_tensor, rot_target = self._generate_rotnet_data(img)

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
        rot_img = self._basic_square_crop(img)
        rot_img = self._resize(rot_img)
        image_tensor = self._to_tensor(rot_img)

        # generate rotation angle
        theta = np.random.choice(self.rot_deg, size=1)[0]
        rotated_image_tensor = self.rotate_image(image_tensor.unsqueeze(0), theta).squeeze(0)

        # produce label
        target = torch.zeros(self.num_rotations)
        target[self.rot_deg.index(theta)] = 1

        return rotated_image_tensor, target
    
    def rotate_image(self, image_tensor, theta):
        """
        Detials
        """
        # get tensor image data type
        dtype = image_tensor.dtype
        theta *= np.pi/180
        theta = torch.tensor(theta)

        rotation_matrix = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                                        [torch.sin(theta), torch.cos(theta), 0]])
        rotation_matrix = rotation_matrix[None, ...].type(dtype).repeat(image_tensor.shape[0], 1, 1)
        
        grid = F.affine_grid(rotation_matrix,
                                     image_tensor.shape,
                                     align_corners=True).type(dtype)
        rotated_torch_image = F.grid_sample(image_tensor, grid, align_corners=True)

        # returning rotated image tensor
        return rotated_torch_image

    def _to_tensor(self, img):
        transform = T.ToTensor()
        ten_img = transform(img)
        return ten_img
    
    def __len__(self):
        return len(self.ids)
    
    def _basic_square_crop(self, img):
        """ Detials """

        width, height = img.size
        centre_width = width/2
        centre_height = height/2
        max_size = min(width, height)
        half_max = max_size/2
        left = centre_width - half_max
        right = centre_width + half_max
        top = centre_height - half_max
        bottom = centre_height + half_max
        cropped_img = img.crop((left, top, right, bottom))

        return cropped_img
    
    def _resize(self, img, size=1000):
        """ Detials """
        resized_img = img.resize((size, size))
        return(resized_img)
  
def COCO_collate_function(batch):
    """Detials"""
    return tuple(zip(*batch)) 