"""
Title:      COCODataset.py

Fuction:    The script is the location for all elements of loading datasets into the
            tuber segmentation pipeline. this includes the functions used to structure the datsets
            for use in the model and for functions that assist that load the data into the models
            for use. 

Edited by:  Bradley Hurst 
"""
# ============================
# Importing libraries/packages
# ============================  
# torch imports
import json
import torch
import torchvision.transforms as T
import torch.utils.data as data
import torch.nn.functional as F

# supporting package imports
import os 
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from PIL import Image
import numpy as np

#import transforms as T
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# utils
from utils import basic_square_crop, resize

# ============================
# Classes and functions for data loading
# ============================
class COCORotDataset(data.Dataset):
    """
    Title:      COCOLoader

    Function:   The class inherits from the torch.utils.data.Dataset and modifies the __getitem__  
                and __len__ methods for the function 

    Inputs:     - json string
                - image dir string 
                - transforms
                
    Outputs:    - a image tensor and target tensor when __getitem__ method is called
                - a id length value when __rep__ is called

    Deps:

    Edited by:  Bradley Hurst
    """
    def __init__(self, root, json_root, transforms=None, train=False, num_rotations=4, seed=42):
        self.root = root
        self.coco = COCO(json_root)
        self.ids = list(self.coco.imgs.keys())
        self.transforms = transforms
        self.train = train

        # rotnet stuff
        self.rotation_degrees = np.linspace(0, 360, num_rotations + 1).tolist()[:-1]
        self.num_rotations = num_rotations
        self.seed = seed

    def __getitem__(self, idx):
        
        # getting ids for specific image
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds = img_id)
        
        # loading annotations for annotation ids
        anns = self.coco.loadAnns(ann_ids)

        # initialisng lists from target data
        labels = []
        boxes = []        
        masks_list = []
        areas = []
        iscrowds = []
        
        # itterating though loaded anns
        for ann in anns:
            
            # collecting data labels and areas for target
            labels.append(ann['category_id'])
            areas.append(ann['area'])

            # formatting and collecting bbox data 
            bbox = ann['bbox']            
            new_bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
            boxes.append(new_bbox)

            # formatting and collecting iscrowd id's
            if ann["iscrowd"]:
                iscrowds.append(1)
            else:
                iscrowds.append(0)

            # formatting mask to tensor and collecting masks
            mask = self.coco.annToMask(ann)
            mask == ann['category_id']
            masks_list.append(torch.from_numpy(mask))

        # converting lists to tensors for target
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = torch.as_tensor(areas, dtype=torch.int64)
        masks = torch.stack(masks_list, 0)
        iscrowd = torch.as_tensor(iscrowds, dtype=torch.int64)
        image_id = torch.tensor([idx])

        # assembling target dict
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        # laoding image
        image_path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, image_path)).convert('RGB')
        rot_img = img
        
        # converting to tensor
        im_conv = T.ToTensor()
        img = im_conv(img)
        
        ###########################################################################
        #if self.train == True:
        #    augs = A.Compose([
        #        #A.Normalize(),
        #        A.Downscale (scale_min=0.8,
        #                     scale_max=0.8,
        #                     interpolation=None,
        #                     p=0.1),
        #        A.OneOf([
        #            A.HueSaturationValue(hue_shift_limit=0.5,
        #                                 sat_shift_limit= 0.5, 
        #                                 val_shift_limit=0.5, 
        #                                 p=0.3),
        #            A.RandomBrightnessContrast(brightness_limit=0.5, 
        #                                       contrast_limit=0.5,
        #                                       p=0.3),
        #            ],p=0.2),
        #        A.ToGray(p=0.6),
        #        ToTensorV2()
        #        ], p=1)
        #    pil_trans = T.ToPILImage()
        #    p_img = pil_trans(img)
        #    np_img = np.array(p_img)
        #    img = augs(image=np_img)['image']
        ###########################################################################

        # applying transforms if applicable
        if self.transforms != None:
            img, target = self.transforms(img, transform)

        # === ROTNET ========================================================== #
        # getting basic image square
        rot_img = basic_square_crop(rot_img)
        rot_img = resize(rot_img)

        # tensor conversion
        rot_img = im_conv(rot_img)

        # select random rotation
        theta = np.random.choice(self.rotation_degrees, size=1)[0]
        rot_img = self.rotate_image(rot_img.unsqueeze(0), theta).squeeze(0)

        label = torch.zeros(self.num_rotations)
        label[self.rotation_degrees.index(theta)] = 1

        return img, target, rot_img, label

    def rotate_image(self, image_tensor, theta):
        """
        Detials
        """
        # get tensor image data type
        dtype = image_tensor.dtype

        # covert degrees to radians and converting to tensor
        theta *= np.pi/180
        theta = torch.tensor(theta)

        # retrieveing rotation matrix around the z axis
        rotation_matrix = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                                        [torch.sin(theta), torch.cos(theta), 0]])
        rotation_matrix = rotation_matrix[None, ...].type(dtype).repeat(image_tensor.shape[0], 1, 1)
        
        # appling rotation
        grid = F.affine_grid(rotation_matrix,
                                     image_tensor.shape,
                                     align_corners=True).type(dtype)
        rotated_torch_image = F.grid_sample(image_tensor, grid, align_corners=True)

        # returning rotated image tensor
        return rotated_torch_image
    
    def __len__(self):
        return len(self.ids)

def COCO_collate_function(batch):
    """
    Title:      collate_function

    Function:   function formats how batch is formated

    Inputs:     batch
                
    Outputs:    formatted batch

    Deps:

    Edited by:  Bradley Hurst
    """
    return tuple(zip(*batch)) 

    ## before testing, all this will need to be tested when on the web
    #images = torch.stack(images, 0)

# test
if __name__ == "__main__":

    dataset = COCODataset("sources/jersey_dataset_v4/train",
                         "sources/jersey_dataset_v4/train/train.json")
    
    dataloader = torch.utils.data.DataLoader(dataset,
                                        batch_size = 2,
                                        shuffle = False,
                                        num_workers = 4,
                                        collate_fn = COCO_collate_function)

    data = next(iter(dataloader))

    x, y = data

    print(x)  
    print(y)  
