"""
Module Detials:
This module is a high level implementations of the training process for
deep learning models. The train class which is imported by the main file
uses the provided config dictionary to initialise the other modules used 
for training. To execute model training the train method is called from 
the main file.
"""
# imports
# base packages
import random

# third party packages
import torch
import numpy as np

# local packages
from data_handling import Loaders

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import cv2

# loader_cfg
loader_cfg = {
    "source": "data_handling/sources/jersey_dataset_v4",
    "random_seed": 42,
    "model_name": "mask_rcnn",
    "col_fn": True,
    "params":{
        "train":{
            "dir": "train",
            "json": "train.json",
            "batch_size": 1,
            "shuffle": True,
            "num_workers": 0,
            "augmentations": False},
        "val":{
            "dir": "val",
            "json": "val.json",
            "batch_size": 1,
            "shuffle": False,
            "num_workers": 0,
            "augmentations": False},
        "test":{
            "dir": "test",
            "json": "test.json",
            "batch_size": 1,
            "shuffle": False,
            "num_workers": 0,
            "augmentations": False}}
}

def overlay_masks(image, target):
    """
    Overlay masks on the image.

    :param image: Image tensor.
    :param target: Target dictionary containing masks and other annotations.
    :return: PIL Image with masks overlaid.
    """
    # Convert the tensor image to PIL for easy manipulation
    image = image.squeeze().permute(1, 2, 0).numpy()
    image = Image.fromarray((image * 255).astype(np.uint8))

    # Draw masks
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for i in range(len(target["masks"])):
        mask = target["masks"][i].squeeze().numpy()
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            polygon = patches.Polygon(contour.reshape(-1, 2), linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(polygon)

    # Save the figure to an image
    fig.savefig(str(i) + "_overlayed_image.png", bbox_inches='tight', pad_inches=0)
    plt.close(fig)

if __name__ == "__main__":
    # get loader
    train_loader, val_loader = Loaders(loader_cfg, "train").loader()

    val_iter = iter(val_loader)

    for i in range(len(val_loader)):
        img, target = next(val_iter)

        overlay_masks(img[0], target[0])