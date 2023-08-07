"""
Title:      COCODataset.py
Function:   Responsible for loading datasets into the segmentation pipeline. Contains utilities to structure datasets 
            and to load them into models.
Edited by:  Bradley Hurst
"""

# ============================
# Importing libraries/packages
# ============================
import os
import json
import torch
import numpy as np
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from utils import basic_square_crop, resize

# ============================
# Dataset and utilities
# ============================
class COCORotDataset(Dataset):
    def __init__(self, root, json_root, transforms=None, train=False, num_rotations=4, seed=42):
        self.root = root
        self.coco = COCO(json_root)
        self.ids = list(self.coco.imgs.keys())
        self.transforms = transforms
        self.train = train
        self.rotation_degrees = np.linspace(0, 360, num_rotations + 1).tolist()[:-1]
        self.num_rotations = num_rotations
        self.seed = seed

    def __getitem__(self, idx):
        # Retrieve image and annotations
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # Process annotations
        labels, boxes, masks_list, areas, iscrowds = [], [], [], [], []
        for ann in anns:
            labels.append(ann['category_id'])
            areas.append(ann['area'])
            new_bbox = [ann['bbox'][0], ann['bbox'][1], ann['bbox'][0]+ann['bbox'][2], ann['bbox'][1]+ann['bbox'][3]]
            boxes.append(new_bbox)
            iscrowds.append(int(ann["iscrowd"]))

            mask = self.coco.annToMask(ann)
            mask == ann['category_id']
            masks_list.append(torch.from_numpy(mask))

        # Assemble target dictionary
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "masks": torch.stack(masks_list, 0),
            "image_id": torch.tensor([idx]),
            "area": torch.as_tensor(areas, dtype=torch.int64),
            "iscrowd": torch.as_tensor(iscrowds, dtype=torch.int64)
        }

        # Load and process image
        image_path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, image_path)).convert('RGB')
        rot_img = basic_square_crop(img)
        rot_img = resize(rot_img)

        # Randomly rotate for RotNet
        theta = np.random.choice(self.rotation_degrees, size=1)[0]
        rot_img = self.rotate_image(T.ToTensor()(rot_img).unsqueeze(0), theta).squeeze(0)

        label = torch.zeros(self.num_rotations)
        label[self.rotation_degrees.index(theta)] = 1

        # Convert img to tensor and apply transforms
        img = T.ToTensor()(img)
        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target, rot_img, label

    def rotate_image(self, image_tensor, theta):
        dtype = image_tensor.dtype
        theta *= np.pi/180
        rotation_matrix = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                                        [torch.sin(theta), torch.cos(theta), 0]])
        rotation_matrix = rotation_matrix[None, ...].type(dtype).repeat(image_tensor.shape[0], 1, 1)

        grid = F.affine_grid(rotation_matrix, image_tensor.shape, align_corners=True).type(dtype)
        rotated_torch_image = F.grid_sample(image_tensor, grid, align_corners=True)

        return rotated_torch_image

    def __len__(self):
        return len(self.ids)


def COCO_collate_function(batch):
    return tuple(zip(*batch))


# Test Block
if __name__ == "__main__":
    dataset = COCORotDataset("sources/jersey_dataset_v4/train", "sources/jersey_dataset_v4/train/train.json")
    
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=COCO_collate_function)

    data = next(iter(dataloader))
    x, y = data

    print(x)  
    print(y)
