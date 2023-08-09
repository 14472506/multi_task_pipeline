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
from utils import basic_square_crop, resize
import torchvision.transforms as torch_trans
import torch.nn.functional as torch_fun

# class
class RotNetDataset(data.Dataset):
    """
    Detials
    """
    def __init__(self, cfg, seed=42):

        self.root = cfg["source"]
        self.num_rotations= cfg["params"]["num_rotations"]

        # retrieveing image data from root directory
        self.root = os.path.expanduser(self.root)
        self.images = []
        for image in os.listdir(self.root):
            self.images.append(image)

        # setting numpy random seed
        np.random.seed(seed)

        # rotnet stuff
        self.rotation_degrees = np.linspace(0, 360, self.num_rotations + 1).tolist()[:-1]
        self.num_rotations = self.num_rotations
        self.seed = seed

    def __getitem__(self, idx):
        """
        method_name : __getitem__

        task        : base method that returnes indexed image from dataset when called

        edited by   : bradley hurst
        """
        # load called RGB image 
        img_path = os.path.join(self.root, self.images[idx])
        try: 
            image = Image.open(img_path).convert("RGB")
        except OSError:
            print("ERROR")
            print(img_path)

        # getting basic image square
        image = basic_square_crop(image)
        image = resize(image)

        # converting image to tensor
        tensor_transform = torch_trans.Compose([torch_trans.ToTensor()])
        image_tensor = tensor_transform(image)

        # select random rotation
        theta = np.random.choice(self.rotation_degrees, size=1)[0]
        rotated_image_tensor = self.rotate_image(image_tensor.unsqueeze(0), theta).squeeze(0)
        #label = torch.tensor(self.rotation_degrees.index(theta)).long()

        label = torch.zeros(self.num_rotations)
        label[self.rotation_degrees.index(theta)] = 1

        # returning rotated image tensor and label
        return rotated_image_tensor, label

    def __len__(self):
        """
        Details
        """
        return len(self.images)

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
        grid = torch_fun.affine_grid(rotation_matrix,
                                     image_tensor.shape,
                                     align_corners=True).type(dtype)
        rotated_torch_image = torch_fun.grid_sample(image_tensor, grid, align_corners=True)

        # returning rotated image tensor
        return rotated_torch_image