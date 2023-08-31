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
import torchvision.transforms as T
import torch.nn.functional as F

# class
class RotNetDataset(data.Dataset):
    """
    Detials
    """
    def __init__(self, cfg, seed=42):
        """ Detials """
        self.cfg = cfg
        self._exctract_config()
        self._set_seed()
        self._initialise_params()
        self._get_images()

    def __getitem__(self, idx):
        """ Detials """
        # get image 
        img_path = os.path.join(self.root, self.images[idx])
        try: 
            image = Image.open(img_path).convert("RGB")
        except OSError:
            pass
        
        # image to tensor
        image = self._basic_square_crop(image)
        image = self._resize(image)
        tensor_transform = T.Compose([T.ToTensor()])
        image_tensor = tensor_transform(image)

        # select random rotation
        theta = np.random.choice(self.rotation_degrees, size=1)[0]
        rotated_tensor = self._rotate_image(image_tensor.unsqueeze(0), theta).squeeze(0)

        # produce label
        label = torch.zeros(self.num_rotations)
        label[self.rotation_degrees.index(theta)] = 1

        # returning rotated image tensor and label
        return rotated_tensor, label

    def __len__(self):
        """ Details """
        return len(self.images)
    
    def _exctract_config(self):
        """ Detials """
        self.root = self.cfg["source"]
        self.seed = self.cfg["random_seed"]
        self.num_rotations= self.cfg["params"]["num_rotations"]

    def _set_seed(self):
        """ Detials """
        np.random.seed(self.seed)

    def _get_images(self):
        """ Detials """
        self.root = os.path.expanduser(self.root)
        self.images = []
        for image in os.listdir(self.root):
            self.images.append(image)
    
    def _initialise_params(self):
        """ Detials """
        self.rotation_degrees = np.linspace(0, 360, self.num_rotations + 1).tolist()[:-1]

    def _rotate_image(self, image_tensor, theta):
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