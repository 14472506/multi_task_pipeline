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
from utils import basic_square_crop, resize, jigsaw_permuatations
import torchvision.transforms as torch_trans
import torch.nn.functional as torch_fun

class JigsawDataset(data.Dataset):
    """
    Detials
    """

    def __init__(self, root, num_tiles=9, num_permutations=1000, permgen_method='maximal',
                 grayscale_probability=0.3, buffer=True, jitter=True, normalization=True, seed=42):
        """
        Detials
        """
        # retrieveing image data from root directory
        self.root = os.path.expanduser(root)
        self.images = []
        for image in os.listdir(self.root):
            self.images.append(image)

        # setting numpy random seed
        np.random.seed(seed)

        # number of tiles and permutations
        self.num_tiles = num_tiles
        self.num_permutations = num_permutations
        self.perm_method = permgen_method
        
        # buffer for image processing
        self.buffer = buffer

        # image modification
        self.grayscale_probability = grayscale_probability
        self.jitter = jitter
        self.normalize = normalization

        #self.permutations = self.generate_permutation_set(num_tiles = self.num_tiles,
        #                                                  num_permutations = self.num_permutations,
        #                                                  method = self.perm_method)

        self.permutations = jigsaw_permuatations(str(num_permutations))

    def __getitem__(self, idx):
        """
        Detials
        """
        # load called RGB image 
        img_path = os.path.join(self.root, self.images[idx])
        image = Image.open(img_path).convert("RGB")

        # getting basic image square
        image = basic_square_crop(image)
        image = resize(image)

        # getting tile construction data from image
        width, _= image.size
        num_tiles_per_dimension = int(np.sqrt(self.num_tiles))
        tile_length = width // num_tiles_per_dimension 

        # converting image to tensor
        tensor_transform = torch_trans.Compose([torch_trans.ToTensor()])
        image_tensor = tensor_transform(image)
    
        # data collection and buffer init
        tiles = []
        buffer = int(tile_length * 0.1)

        # entering set of loops to go through both dimensions of image
        for i in range(num_tiles_per_dimension):
            for j in range(num_tiles_per_dimension):
                if self.buffer:
                    tile_ij = torch.empty(image_tensor.shape[0], tile_length - buffer, tile_length - buffer)
                    buffer_x1, buffer_x2 = np.random.multinomial(buffer, [0.5, 0.5])
                    buffer_y1, buffer_y2 = np.random.multinomial(buffer, [0.5, 0.5])
                    tile_x1 = i * tile_length + buffer_x1 
                    tile_x2 = (i + 1) * tile_length - buffer_x2
                    tile_y1 = j * tile_length + buffer_y1 
                    tile_y2 = (j + 1) * tile_length - buffer_y2
                    tile_ij = image_tensor[:, tile_x1: tile_x2, tile_y1: tile_y2]
                else:
                    tile_ij = image_tensor[:,
                                i * tile_length: (i + 1) * tile_length,
                                j * tile_length: (j + 1) * tile_length]

                tiles.append(tile_ij)

        # Tensorising tiles
        tiles = torch.stack(tiles)
               
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