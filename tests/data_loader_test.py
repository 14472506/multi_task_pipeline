"""
Detials
"""
# imports
from datasets import DataloaderBuilder
import numpy as np
import time
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# classes 
class LoaderTest():
    """ Detials """
    def __init__(self, cfg, loader_type):
        """Detials"""
        self.cfg = cfg
        self.loader_type = loader_type
        self._initialise_loader()
        self.to_img = T.ToPILImage()

    def _initialise_loader(self):
        """Detials"""
        self.loader1, self.loader2 = DataloaderBuilder(self.cfg, "train").loader()
        print(len(self.loader1))
        print(len(self.loader2))
    
    def test_loader(self):
        """Detials"""
        for i, data in enumerate(self.loader1):

            print(data)
            s_tens, s_targ, ssl_tens, ssl_targ = data
            #print(s_targ)

            image = s_tens[0].numpy()
            img_arr = self.to_img(image)

            masks_list = []
            boxes_list = []

            for mask, boxes in zip(s_targ[0]["masks"], s_targ[0]["boxes"]):
                mask_img = self.to_img(mask)
                mask_arr = np.array(mask_img)
                masks_list.append(mask_arr)
                box = boxes.tolist()
                boxes_list.append(box)

            colours = np.random.randint(0, 255, size=(len(masks_list), 3))
            overlay =  np.zeros_like(img_arr)

            for masks, colours in zip(masks_list, colours):
                overlay[masks == 1] = colours

            combined_img = np.where(overlay > 0, overlay, img_arr)

            fig, ax = plt.subplots(1)
            ax.imshow(combined_img)
            for box in boxes_list:
                x, y, w, h = box
                rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

            plt.savefig("output_img.png")   # Save the current figure to a file
            plt.close()   

            time.sleep(2)

