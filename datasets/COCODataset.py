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

# supporting package imports
import os 
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from PIL import Image
import numpy as np

#import transforms as T
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# ============================
# Classes and functions for data loading
# ============================
class COCODataset(data.Dataset):
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
    def __init__(self, root, json_root, transforms=None, train=False):
        self.root = root
        self.coco = COCO(json_root)
        self.ids = list(self.coco.imgs.keys())
        self.transforms = transforms
        self.train = train

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

        return img, target
    
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
