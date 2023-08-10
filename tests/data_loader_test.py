"""
Detials
"""
# imports
from datasets import DataloaderBuilder
import numpy as np
import matplotlib.pyplot as plt
import time

# classes 
class LoaderTest():
    """ Detials """
    def __init__(self, cfg, loader_type):
        """Detials"""
        self.cfg = cfg
        self.loader_type = loader_type
        self._initialise_loader()

    def _initialise_loader(self):
        """Detials"""
        self.loader = DataloaderBuilder(self.cfg, "train").loader()
    
    def test_loader(self):
        """Detials"""
        for i, data in enumerate(self.loader):

            s_tens, s_targ, ssl_tens, ssl_targ = data

            image = s_tens[0].numpy()

            plt.imsave("output_img.png", image)

            time.sleep(2)


            ## Display the s_tens and ssl_tens images with their respective targets.
            ## Assuming batch size > 1, this will display the first image in the batch.
            #fig = plt.figure(figsize=(12, 6))
            #
            #ax1 = fig.add_subplot(1, 2, 1)
            #ax1.set_title(f"s_targ: {s_targ[0]}")
            #self._imshow(s_tens[0])
            #
            #ax2 = fig.add_subplot(1, 2, 2)
            #ax2.set_title(f"ssl_targ: {ssl_targ[0]}")
            #self._imshow(ssl_tens[0])
            #
            ## This is to just display a few batches and then break. You can adjust the number.
            #if i == 3: 
            #    break

    def _imshow(self, tensor, title=None):
        image = tensor.numpy().transpose((1, 2, 0))
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        image = std * image + mean
        image = np.clip(image, 0, 1)
        plt.imshow(image)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)
