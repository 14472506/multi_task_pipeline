# imports
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from models import ModelBuilder

import os
import numpy as np
import cv2
import random

# vsiualiser function
class Visualiser():
    """
    Detials
    """
    def __init__(self, weights_path, weight_name, imgs_root):
        """
        Detials
        """

        self.config = {
            "model_name": "Mask_RCNN_Resnet_50_FPN",
            "pre_trained": False,
            "loaded": False,
            "load_source": "",
            "trainable_layers": 5,
            "num_classes": 2
        }

        self.model = ModelBuilder(self.config).model_builder()
        self.image_root = imgs_root
        self.weights_path = weights_path
        self.weights_name = weight_name

    def load_model(self, path, pth_name, model):
        """
        Detials
        """
        # load best model so far
        load_path = path + "/" + pth_name
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint["state_dict"])

    def get_mask(self, image, mask, colour, alpha = 0.5):
        """
        Details
        """
        # Create a copy of the input image
        blended_image = image.copy()

        # Apply the mask and blend it with the original image using the alpha value
        for c in range(3):
            blended_image[:, :, c] = np.where(mask > 0.5, image[:, :, c] * (1 - alpha) + colour[c] * alpha, image[:, :, c])

        return blended_image
    
    def main(self):
        """
        Detials        
        """
        # load weights
        self.load_model(self.weights_path,
            self.weights_name,
            self.model)
        
        # set model to eval 
        self.model.Mask_RCNN.eval()
        
        # define image tranfrom
        transfom = transforms.Compose([
            transforms.ToTensor(),
        ])

        # instance threshold
        thresh = 0.5

        count = 1
        # begin image loop
        for filename in os.listdir(self.image_root):
            if filename.endswith(".jpg"):

                # image path
                image_path = os.path.join(self.image_root, filename)

                # load image
                image = Image.open(image_path)
                np_img = np.array(image)

                # image tensor
                img_tensor = transfom(image)
                img_tensor = img_tensor.unsqueeze(0)

                # get model output
                with torch.no_grad():
                    output = self.model([img_tensor])
                
                # outputs
                masks = output[0]['masks'].numpy()
                labels = output[0]['labels'].numpy()
                scores = output[0]['scores'].numpy()
        
                # apply masks from instances
                for i in range(masks.shape[0]):
                    if scores[i] > thresh:
                        mask = masks[i][0]
                        label = labels[i]
                        colour = np.random.randint(0, 256, 3)
                        np_img = self.get_mask(np_img, mask, colour)


                title = "out_img_" + str(count) + ".jpg"
                img =  cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(title, img)
                count += 1

if __name__=="__main__":
    
    im_path = "datasets/sources/jersey_dataset_v4/test"
    mode_name = "best.pth"
    mod_path =  "outputs/reduced_dataset_ssl/full_reduction_rotnet_pt_mask_rcnn_5"
    vis = Visualiser(mod_path, mode_name, im_path)
    vis.main()