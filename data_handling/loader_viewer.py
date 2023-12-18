# Imports
import torch
from data_handling import Loaders
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt
import random

# loader cfg
loader_cfg = {
    "source": "data_handling/sources/All_RGB",
    "random_seed": 42,
    "model_name": "rotnet_resnet_50",
    "col_fn": None,
    "params": {
        "col_fn": None,
        "num_rotations": 4,
        "split":{
            "train_test": 0.8,
            "train_val": 0.8,
        },
        "train":{
            "batch_size": 1,
            "shuffle": True,
            "num_workers": 0,
            "augmentations": True
        },
        "val": {
            "batch_size": 1,
            "shuffle": False,
            "num_workers": 0,
            "augmentations": False
        },
        "test": {
            "batch_size": 1,
            "shuffle": False,
            "num_workers": 0,
            "augmentations": False
        },
    },
}

# get loader
train_loader, _ = Loaders(loader_cfg, "train").loader()

# get iterator
loader_iterator = iter(train_loader)

for i in range(20):
    sup_im, sup_target = next(loader_iterator)

# Create a figure with 3x3 grid of subplots
#fig, axes = plt.subplots(3, 3, figsize=(10, 10))
#axes = axes.ravel()  # Flatten the array of axes

for j in range(sup_im.size(0)):
    count = 0
    #for i in range(sup_im[0].size(0)):
    #    image = sup_im[j][i, :, :, :]
    
    # Convert image tensor from CHW to HWC format and normalize
    image = sup_im[j].permute(1, 2, 0)  # CHW to HWC
    image = (image - image.min()) / (image.max() - image.min())  # Normalize
    image = (image * 255).byte().numpy()  # Convert to 8-bit and to numpy array

    ## Display image in the corresponding subplot
    #axes[count].imshow(image)
    #axes[count].axis('off')  # Turn off axis
    #axes[count].set_title(f"TILE_{count}")
    #count += 1

    name = str(j) + "_auged_img.png"

    # Adjust layout and save the figure as a single image
    plt.tight_layout()
    plt.savefig(name)