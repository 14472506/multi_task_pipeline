"""
Something Something Something
"""
# imports
# base packages 
import os
from PIL import Image
import json

# thirst party packages
import numpy as np
import torch
import torch
import torch.utils.data as data
import torchvision.transforms as T
import torch.utils.data as data
import torch.nn.functional as F
from pycocotools.coco import COCO

# local packages

# class
class COCOJigsawDataset(data.Dataset):
    """ Detials """
    def __init__(self, cfg, type, seed=42):
        self.cfg = cfg
        self.type = type
        self._extract_config()
        self._init_params()

    def _extract_config(self):
        """ Details """
        self.sup_source = self.cfg["source"][0]
        self.num_tiles = self.cfg["params"]["num_tiles"]
        self.num_perms = self.cfg["params"]["num_permutations"]
        self.type_dir = self.cfg["params"][self.type]["dir"]
        self.type_json = self.cfg["params"][self.type]["json"]
    
    def _init_params(self):
        """ Detials """
        self.permutations = jigsaw_permuatations(self.num_perms)
        self.coco = COCO(os.path.join(self.sup_source, self.type_dir, self.type_json))
        self.dir = os.path.join(self.sup_source, self.type_dir)
        self.ids = list(self.coco.imgs.keys())

    def __getitem__(self, idx):
        """ Detials """
        # laoding image and related coco data
        img_id = self.ids[idx]
        file_name = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.dir, file_name)).convert('RGB')
        width, height = img.size
        img_tensor = self._to_tensor(img)

        # select random permutation and generate rotation ground truth
        permutation_index = np.random.randint(0, self.num_perms)
        permutation = torch.tensor(self.permutations[permutation_index])
        stack_target = torch.tensor(permutation_index)

        # get instance ground truth
        mrcnn_targets = self._coco_target_collection(img_id, idx)

        # apply permutations to image and mask data
        im_stack, perm_targets = self._apply_permutations(img_tensor, mrcnn_targets, width, height, permutation)

        return im_stack, perm_targets, stack_target
        
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
    
    def _apply_permutations(self, img_tensor, instance_targets, width, height, permutation):
        """ Detials """
        # process image instance targets
        masks = instance_targets["masks"]
        
        # and image to to masks
        all_tensors = torch.cat((img_tensor.unsqueeze(0), masks.unsqueeze(0)), dim=1)

        # split tensor into tiles
        tiles = self._tensor_to_tiles(all_tensors, width, height)
        tiles[:, :, :, :] = tiles[permutation, :, :, :]

        tensor_dim = tiles.size(1)
        img_stack, targets_stack = torch.split(tiles, [3, tensor_dim - 3], dim=1)

        perm_masks, perm_boxes, perm_labels = self._perm_corrector(targets_stack)

        #perm_masks = perm_masks.permute(1, 0, 2, 3)
        #img_stack = img_stack.permute(1, 0, 2, 3)

        # add updated instances anns to target data
        instance_targets["masks"] = perm_masks
        instance_targets["boxes"] = perm_boxes
        instance_targets["labels"] = perm_labels

        return img_stack, instance_targets

    def _tensor_to_tiles(self, tensor, width, height):
        """ Detials """
        # get tile constructors
        num_tiles_per_dimension = int(np.sqrt(self.num_tiles))
        width_tiles = width // num_tiles_per_dimension
        height_tiles = height // num_tiles_per_dimension

        tensor = tensor.squeeze(0)

        tiles = []
        for i in range(num_tiles_per_dimension):
            for j in range(num_tiles_per_dimension):
                
                hmin =  i * height_tiles
                hmax = (i+1) * height_tiles
                wmin =  j * width_tiles
                wmax = (j+1) * width_tiles

                tile_ij = tensor[:, hmin: hmax, wmin: wmax]

                tiles.append(tile_ij)
        tiles = torch.stack(tiles)

        return tiles
    
    def _perm_corrector(self, targets_stack):
        """ Detials """
        # init lists to hold the restructured target data
        perm_masks = []
        perm_boxes = []
        perm_labels = []
        
        # look through current targets
        for i in range(targets_stack.size(1)):
            # select a layer for processing
            layer = targets_stack[:, i, :, :]
            # init list for collecting masked tile indeces
            mask_indices=[]

            # check for tile masks
            for j in range(layer.size(0)):
                tile = layer[j, :, :]
                if tile.any():
                    mask_indices.append(j)

            # if only one tile containes mask
            if len(mask_indices) == 1:
                # collect mask tiles layer
                perm_masks.append(layer.unsqueeze(1))
                # collect boxes for layer
                boxes = self._get_boxes(layer, mask_indices[0])
                perm_boxes.append(boxes)               
                # collect tile label
                label = [0]*9
                label[mask_indices[0]] = 1
                perm_labels.append(label)

            else :
                for idx in mask_indices:
                    # init replacement layer
                    new_layer = torch.zeros_like(layer)
                    new_layer[idx, :, :] = layer[idx, :, :]
                    perm_masks.append(new_layer.unsqueeze(1))              
                    # collect boxes for layer
                    boxes = self._get_boxes(new_layer, idx)
                    perm_boxes.append(boxes)
                    # collect tile label
                    label = [0]*9
                    label[idx] = 1
                    perm_labels.append(label)
        
        perm_mask = torch.cat(perm_masks, dim=1)
        perm_boxes = torch.as_tensor(perm_boxes, dtype=torch.float32).squeeze(1)
        perm_labels = torch.as_tensor(perm_labels, dtype=torch.int64)

        return perm_mask, perm_boxes, perm_labels
    
    def _get_boxes(self, layer, idx):
        """ Detials """
        # get specific mask tile
        mask = layer[idx, :, :]

        # get limits
        amin = 0

        binary_mask = (mask > 0.5).float()
        non_zero_y, non_zero_x = binary_mask.nonzero(as_tuple=True)

        ymin, _ = torch.min(non_zero_y, dim=0)
        xmin, _ = torch.min(non_zero_x, dim=0)
        ymax, _ = torch.max(non_zero_y, dim=0)
        xmax, _ = torch.max(non_zero_x, dim=0)
        xmin, ymin, xmax, ymax = xmin.item(), ymin.item(), xmax.item(), ymax.item()

        if xmin == xmax:
            if xmin-1 <= amin:
                xmax += 1
            else:
                xmin -= 1

        if ymin == ymax:
            if ymin-1 <= amin:
                ymax += 1
            else:
                ymin -= 1
            
        bbox = [xmin, ymin, xmax, ymax]

        return bbox

    def _mask_corrector(self, mask_stack):
        """ Detials """


        new_stack = torch.cat(new_stack, dim=1)

        labels_list = [1]*new_stack.size(1)
        labels = torch.as_tensor(labels_list, dtype=torch.int64)

        return new_stack, labels
    
    def _box_corrector(self, masks):
        """ Detials """
        # get limits
        amin = 0
        hmax = masks.size(1)
        wmax = masks.size(2)

        bboxes = []
        for mask in masks:
            binary_mask = (mask > 0.5).float()

            non_zero_y, non_zero_x = binary_mask.nonzero(as_tuple=True)

            ymin, _ = torch.min(non_zero_y, dim=0)
            xmin, _ = torch.min(non_zero_x, dim=0)
            ymax, _ = torch.max(non_zero_y, dim=0)
            xmax, _ = torch.max(non_zero_x, dim=0)

            xmin, ymin, xmax, ymax = xmin.item(), ymin.item(), xmax.item(), ymax.item()

            if xmin == xmax:
                if xmin-1 <= amin:
                    xmax += 1
                else:
                    xmin -= 1
            
            if ymin == ymax:
                if ymin-1 <= amin:
                    ymax += 1
                else:
                    ymin -= 1
            

            bbox = [xmin, ymin, xmax, ymax]
            bboxes.append(bbox)
        
        return bboxes
   
    def _to_tensor(self, img):
        transform = T.ToTensor()
        ten_img = transform(img)
        return ten_img

    def __len__(self):
        return len(self.ids)

def jigsaw_permuatations(perm_flag):
    """
    Detials
    """
    if perm_flag == 10:
        ten_perm = ten_perm = [(7, 2, 1, 5, 4, 3, 6, 0, 8), (0, 1, 2, 3, 5, 4, 7, 8, 6),
             (2, 3, 0, 1, 7, 8, 4, 6, 5), (3, 4, 5, 0, 8, 6, 1, 7, 2), (4, 5, 6, 8, 0, 7, 2, 1, 3), 
             (5, 6, 8, 7, 1, 0, 3, 2, 4), (6, 8, 7, 4, 2, 1, 5, 3, 0), (8, 7, 4, 6, 3, 2, 0, 5, 1), 
             (0, 1, 2, 3, 6, 7, 8, 5, 4), (1, 0, 3, 2, 6, 5, 8, 4, 7)]
        return ten_perm
    elif perm_flag == 24:
        twenty_four_perm = [(3, 2, 1, 0), (3, 2, 0, 1), (3, 1, 2, 0), (3, 1, 0, 2), (3, 0, 1, 2), 
                            (3, 0, 2, 1), (2, 3, 1, 0), (2, 3, 0, 1), (2, 1, 3, 0), (2, 1, 0, 3), 
                            (2, 0, 1, 3), (2, 0, 3, 1), (1, 3, 2, 0), (1, 3, 0, 2), (1, 2, 3, 0), 
                            (1, 2, 0, 3), (1, 0, 2, 3), (1, 0, 3, 2), (0, 3, 2, 1), (0, 3, 1, 2), 
                            (0, 2, 3, 1), (0, 2, 1, 3), (0, 1, 2, 3), (0, 1, 3, 2)]
        return twenty_four_perm
    elif perm_flag == 100:
        hundred_perm = [(8, 2, 4, 5, 6, 1, 7, 3, 0), (0, 1, 2, 3, 4, 5, 6, 7, 8),  
            (2, 3, 0, 1, 7, 6, 4, 8, 5), (3, 4, 1, 0, 8, 7, 2, 5, 6), (4, 5, 6, 7, 0, 8, 1, 2, 3), 
            (5, 6, 7, 8, 1, 0, 3, 4, 2), (6, 7, 8, 4, 2, 3, 5, 0, 1), (7, 8, 5, 6, 3, 2, 0, 1, 4), 
            (0, 1, 2, 3, 7, 8, 5, 6, 4), (1, 0, 3, 6, 5, 2, 8, 4, 7), (2, 3, 4, 7, 8, 6, 0, 1, 5), 
            (3, 2, 5, 4, 0, 1, 6, 7, 8), (4, 6, 0, 2, 3, 5, 7, 8, 1), (5, 8, 6, 1, 2, 7, 4, 3, 0), 
            (6, 4, 7, 8, 1, 0, 2, 5, 3), (7, 5, 8, 0, 4, 3, 1, 2, 6), (8, 7, 1, 5, 6, 4, 3, 0, 2), 
            (0, 1, 4, 7, 3, 2, 6, 5, 8), (1, 0, 5, 2, 8, 3, 7, 6, 4), (2, 3, 8, 6, 7, 5, 1, 4, 0), 
            (3, 4, 2, 8, 0, 6, 5, 1, 7), (6, 2, 7, 1, 4, 8, 0, 3, 5), (4, 5, 0, 3, 2, 1, 8, 7, 6), 
            (5, 7, 1, 0, 6, 4, 2, 8, 3), (7, 8, 6, 4, 5, 0, 3, 2, 1), (8, 6, 3, 5, 1, 7, 4, 0, 2), 
            (0, 1, 5, 4, 3, 6, 8, 2, 7), (1, 2, 4, 6, 7, 0, 5, 8, 3), (2, 0, 6, 8, 5, 3, 7, 1, 4), 
            (3, 5, 7, 0, 8, 2, 4, 6, 1), (4, 3, 8, 2, 1, 7, 0, 5, 6), (7, 8, 0, 3, 2, 1, 6, 4, 5), 
            (5, 4, 3, 7, 6, 8, 1, 0, 2), (6, 7, 1, 5, 0, 4, 2, 3, 8), (8, 6, 2, 1, 4, 5, 3, 7, 0), 
            (0, 1, 3, 2, 5, 7, 4, 8, 6), (1, 0, 2, 8, 3, 5, 6, 7, 4), (2, 3, 0, 4, 6, 1, 8, 5, 7), 
            (3, 2, 1, 0, 8, 4, 7, 6, 5), (4, 5, 6, 3, 7, 0, 2, 1, 8), (5, 4, 7, 6, 1, 8, 0, 2, 3), 
            (6, 8, 4, 7, 0, 2, 5, 3, 1), (7, 6, 8, 5, 2, 3, 1, 4, 0), (8, 7, 5, 1, 4, 6, 3, 0, 2), 
            (0, 1, 3, 2, 6, 4, 7, 5, 8), (1, 0, 4, 5, 7, 3, 2, 8, 6), (2, 3, 1, 8, 0, 7, 5, 6, 4), 
            (3, 2, 7, 6, 8, 0, 1, 4, 5), (4, 5, 0, 7, 1, 6, 8, 3, 2), (5, 8, 6, 4, 3, 2, 0, 1, 7), 
            (6, 7, 5, 0, 4, 8, 3, 2, 1), (7, 4, 8, 1, 2, 5, 6, 0, 3), (8, 6, 2, 3, 5, 1, 4, 7, 0), 
            (0, 1, 3, 5, 8, 6, 7, 4, 2), (1, 3, 4, 8, 0, 2, 5, 6, 7), (2, 5, 0, 3, 1, 7, 4, 8, 6), 
            (3, 4, 6, 0, 7, 8, 2, 1, 5), (6, 8, 2, 1, 4, 5, 0, 7, 3), (7, 2, 1, 4, 3, 0, 6, 5, 8), 
            (5, 7, 8, 6, 2, 3, 1, 0, 4), (4, 6, 7, 2, 5, 1, 8, 3, 0), (8, 0, 5, 7, 6, 4, 3, 2, 1), 
            (0, 1, 5, 7, 6, 4, 8, 3, 2), (1, 3, 8, 5, 7, 6, 2, 0, 4), (3, 2, 1, 0, 5, 7, 4, 8, 6), 
            (2, 5, 0, 6, 1, 3, 7, 4, 8), (6, 0, 4, 8, 3, 2, 5, 7, 1), (4, 6, 7, 1, 8, 5, 3, 2, 0), 
            (7, 4, 3, 2, 0, 8, 1, 6, 5), (8, 7, 6, 4, 2, 1, 0, 5, 3), (5, 8, 2, 3, 4, 0, 6, 1, 7), 
            (1, 3, 8, 6, 2, 0, 7, 5, 4), (7, 2, 6, 0, 5, 8, 4, 1, 3), (0, 1, 2, 7, 3, 4, 5, 6, 8), 
            (2, 0, 4, 1, 6, 5, 8, 3, 7), (3, 4, 1, 8, 7, 2, 6, 0, 5), (4, 6, 7, 5, 1, 3, 2, 8, 0), 
            (5, 8, 3, 2, 0, 7, 1, 4, 6), (6, 5, 0, 4, 8, 1, 3, 7, 2), (8, 7, 5, 3, 4, 6, 0, 2, 1), 
            (7, 3, 1, 0, 2, 8, 6, 5, 4), (0, 1, 2, 5, 4, 3, 7, 8, 6), (1, 0, 6, 4, 7, 2, 8, 3, 5), 
            (2, 5, 0, 6, 8, 7, 1, 4, 3), (4, 2, 7, 3, 1, 0, 5, 6, 8), (3, 8, 5, 1, 6, 4, 0, 2, 7), 
            (8, 7, 3, 2, 5, 6, 4, 0, 1), (5, 6, 4, 8, 3, 1, 2, 7, 0), (6, 4, 8, 7, 0, 5, 3, 1, 2), 
            (0, 1, 2, 8, 6, 4, 5, 3, 7), (1, 2, 6, 3, 7, 0, 4, 5, 8), (4, 0, 7, 5, 1, 3, 2, 8, 6), 
            (7, 5, 0, 4, 2, 8, 6, 1, 3), (2, 4, 3, 6, 8, 7, 1, 0, 5), (6, 8, 4, 1, 3, 5, 0, 7, 2), 
            (3, 7, 1, 0, 5, 6, 8, 2, 4), (5, 3, 8, 2, 4, 1, 7, 6, 0), (8, 6, 5, 7, 0, 2, 3, 4, 1), 
            (0, 8, 4, 5, 6, 3, 2, 1, 7), (1, 0, 3, 2, 5, 4, 8, 6, 7)]
        return hundred_perm

def COCO_collate_function(batch):
    """Detials"""
    return tuple(zip(*batch))