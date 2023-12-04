"""
Details
"""
# import
# imports
import os
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as torch_trans
import torch.nn.functional as torch_fun

class JigsawDataset(data.Dataset):
    """
    Detials
    """

    def __init__(self, cfg, seed):
        """
        Detials
        """
        self.cfg = cfg
        self._exctract_config()

        # retrieveing image data from root directory
        self.root = os.path.expanduser(self.root)
        self.images = []
        for image in os.listdir(self.root):
            self.images.append(image)

        #self.permutations = self.generate_permutation_set(num_tiles = self.num_tiles,
        #                                                  num_permutations = self.num_permutations,
        #                                                  method = self.perm_method)
        self.permutations = jigsaw_permuatations(self.num_permutations)

    def _exctract_config(self):
        """ Detials """
        self.root = self.cfg["source"]
        self.seed = self.cfg["random_seed"]
        self.num_tiles= self.cfg["params"]["num_tiles"]
        self.num_permutations = self.cfg["params"]["num_permutations"]
        self.buffer = self.cfg["params"]["buffer"]

    def __getitem__(self, idx):
        """
        Detials
        """
        # load called RGB image 
        img_path = os.path.join(self.root, self.images[idx])
        image = Image.open(img_path).convert("RGB")

        # getting basic image square
        image = self._basic_square_crop(image)
        image = self._resize(image)

        # getting tile construction data from image
        width, height = image.size
        num_tiles_per_dimension = int(np.sqrt(self.num_tiles))
        width_tiles = width // num_tiles_per_dimension
        height_tiles = height // num_tiles_per_dimension

        # converting image to tensor
        tensor_transform = torch_trans.Compose([torch_trans.ToTensor()])
        image_tensor = tensor_transform(image)
        
        # data collection and buffer init
        tiles = self._tensor_to_tiles(image_tensor, width, height)
               
        # randomly shuffle tiles
        y = []

        permutation_index = np.random.randint(0, self.num_permutations)
        permutation = torch.tensor(self.permutations[permutation_index])
        tiles[:, :, :, :] = tiles[permutation, :, :, :]
        y.append(permutation_index)

        # generate ground truth label
        label = torch.zeros(self.num_permutations)
        label[y] = 1

        # return tiles and ground truth label
        return tiles, label 
        
        
    def __len__(self):
        """
        Details
        """
        return len(self.images)
    
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

    def _resize(self, img, size=500):
        """ Detials """
        resized_img = img.resize((size, size))
        return(resized_img)
    
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