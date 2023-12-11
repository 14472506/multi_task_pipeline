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
    
class JigsawWrapper(torch.utils.data.Dataset):
    """ Detials """
    def __init__(self, dataset, transfroms):
        self.dataset = dataset
        self.transforms = transfroms

    def __getitem__(self, idx):
        """ Detials """
        image, label = self.dataset[idx]
        #aug_stack = []
        #
        ## loop through base stack
        #for i in image:
        #    pil_trans = T.ToPILImage()
        #    pil = pil_trans(i)
        #    np_img = np.array(pil)
        #    transformed = self.transforms(image=np_img)["image"]
        #    transformed = torch.tensor(transformed)
        #    transformed = transformed.to(dtype=torch.float32)
        #    aug_stack.append(transformed)
        #
        #stack = torch.stack(aug_stack)
        #stack = stack.permute(0,3,1,2)
        #image = stack
        #
        #return(image, label) 

        num_tiles = image.size(0)
        whole_img = self._tiles_to_tensor(image, num_tiles)

        pil_trans = T.ToPILImage()
        pil = pil_trans(whole_img)
        np_img = np.array(pil)
        
        transformed = self.transforms(image = np_img)

        whole_tensor = torch.from_numpy(transformed["image"])
        whole_tensor = whole_tensor.permute(2, 0, 1)
        whole_tensor = whole_tensor.to(dtype=torch.float32) / 255
        
        image = self._tensor_to_tiles(whole_tensor, num_tiles, whole_tensor.size(2), whole_tensor.size(1))
            
        return(image, label)
    
    def _tiles_to_tensor(self, tensor, num_tiles):
        """ Details """
        num_tiles_per_dimension = int(np.sqrt(num_tiles))
        full_col = None
        count = 0
        for i in range(num_tiles_per_dimension):
            full_row = None
            for j in range(num_tiles_per_dimension):
                if full_row is None:
                    full_row = tensor[count]
                else:
                    full_row = torch.cat((full_row, tensor[count]), dim=2) 
                count += 1
            if full_col is None:
                full_col = full_row
            else:
                full_col = torch.cat((full_col, full_row), dim=1)  
        
        return full_col 
    
    def _tensor_to_tiles(self, tensor, num_tiles, width, height):
        """ Detials """
        # get tile constructors
        num_tiles_per_dimension = int(np.sqrt(num_tiles))
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
        mrcnn_tensor, mrcnn_target, rot_target = self.dataset[idx]

        # converting tensors to arrays
        to_img = T.ToPILImage()        
        mrcnn_img = to_img(mrcnn_tensor)

        mrcnn_arr = np.array(mrcnn_img)

        np_masks = []
        for mask in (mrcnn_target["masks"]): 
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
        
        return mrcnn_transformed, mrcnn_target, rot_target
    
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
        "jigsaw": JigsawWrapper,
        "mask_rcnn": InstanceWrapper,
        "rotmask_multi_task": MultiTaskWrapper,
        "dual_mask_multi_task": InstanceWrapper
    }
    return transform_select[model_type]
