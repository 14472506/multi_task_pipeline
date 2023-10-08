"""
Module Detials:
This module uses a specified model to produced a pseudo labelled 
data which can be used in semi supervised training in a multi task
model training process.
"""
# import
# base packages
import os

# third party packages
from PIL import Image
import torch
from torchvision.transforms import functional as F
import numpy as np
from pycocotools import mask as M
from skimage import measure

# local packages
from models import Models
from loops import Logs

# class
class PseudoLabeller():
    """ Detials """
    def __init__(self, cfg, data_path):
        """ Detials """
        self.cfg = cfg
        self.data_path = data_path

        # extract config
        self._extract_cfg()

        # initialise labelling compononet
        self._initialise_model()
        self._initialise_optimiser()
        self._initialise_logs()
        self._load_model()
        
    def _extract_cfg(self):
        """ Details """
        try:
            self.device = self.cfg["loops"]["device"]
            self.model_cfg = self.cfg["model"]
            self.optimiser_cfg = self.cfg["optimiser"]
            self.logs_cfg = self.cfg["logs"]
        except KeyError as e:
            raise KeyError(f"Missing necessary key in configuration: {e}")

    def _initialise_model(self):
        """
        Initialises the model based off the provided confign
        ensuring the model is sent to the specified device for
        testing
        """
        self.model = Models(self.model_cfg).model()
        self.model.to(self.device)

    def _initialise_optimiser(self):
        """
        Initialises step attribute for edentifying if a scheduler has been
        used in the training process. This is used for determining weather
        pre step of pre and post step models need to be evaluated.
        """
        self.step = True if self.optimiser_cfg["sched_name"] else False

    def _initialise_logs(self):
        """ Initialises the logger module for testing """
        self.logger = Logs(self.logs_cfg)

    def _load_model(self):
        """ Load the trained inference model """
        if self.step:
            model_type = "pre"
        else:
            model_type = "post"
        self.logger.load_model(self.model, model_type)

    def label(self):
        """ Initialising the testing action to be carry out model testing """
        # set model to eval
        self.model.eval()

        # Initialize COCO-style data structure
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        annotation_id = 0

        for filename in os.listdir(self.data_path):
            # get an image from the target directory or skip to the next loop
            if filename.endswith(".jpg"):
                image_path = os.path.join(self.data_path, filename)
                image = Image.open(image_path).convert("RGB")
                input_tensor = F.to_tensor(image).unsqueeze(0)
                input_tensor = input_tensor.to(self.device)

                # get outputs from model
                with torch.no_grad():
                    prediction = self.model(input_tensor)
                outputs = self._filter_predictions(prediction)

                # make coco dataset 
                for i in range(len(outputs["labels"])):
                
                    #annotation_id += 1
                    mask = outputs['masks'][i].numpy()
                    #category_id = int(outputs['labels'][i])
                    #score = float(outputs['scores'][i])
                    #bbox = outputs['boxes'][i].tolist()

                    annotations, area = self._mask_to_polygon(mask)

                    print(annotations, area)

    def _filter_predictions(self, predictions, threshold=0.5):
        """ Filters the predicted masks to get the masks and meta data from the model """
        masks = (predictions[0]["masks"] > threshold).float()
        
        outputs = {
            "masks": masks.detach().cpu(),
            "boxes": predictions[0]["boxes"].detach().cpu(),
            "labels": predictions[0]["labels"].detach().cpu(),
            "scores": predictions[0]["scores"].detach().cpu()
        }

        return(outputs)

    def _mask_to_polygon(self, mask):
        """ 
        requires a mask as an input and returns the polyon, in adition the function also
        returns the area.

        mask input in must be binary and numpy array 
        """
        # area
        fortran_mask = np.asfortranarray(mask)
        encoded_mask = mask.encode(fortran_mask)
        area = mask.area(encoded_mask)

        # annotrations
        contours = measure.find_contours(mask)
        annotations = []
        for contour in contours:
            contour = np.flip(contour, axis=1)
            segmentation = contour.reval().tolist(),
            annotations.append(segmentation)
        
        return annotations, area
            





        # implement this!
        # https://github.com/cocodataset/cocoapi/issues/131

    





            
            



