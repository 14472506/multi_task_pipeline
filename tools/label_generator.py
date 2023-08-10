"""
Detials
"""
# imports
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
import torchvision.transforms as transforms
from torchvision.utils import draw_segmentation_masks
from models import ModelBuilder
from PIL import Image
import os
import json
import numpy as np
import cv2

import gc
import time

# functions
class PseudoLabeller():
    """
    This class provides functionality for loading a pre-trained Mask R-CNN model,
    using it to perform inference on a directory of images, and saving the results
    in COCO format.
    """
    def __init__(self, weight_path, img_dir, out_dir):
        """
        Initializes the PseudoLabeller with the given paths to the model weights,
        image directory, and output directory.
        """
        assert os.path.exists(weight_path), f"Couldn't find weights file at {weight_path}"
        assert os.path.exists(img_dir), f"Couldn't find image directory at {img_dir}"

        self.img_dir = img_dir
        self.out_dir = out_dir
        self.model = self.load_model(weight_path)
        
        self.coco_annotations = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "pseudo_jersey"}]
            }

    def load_model(self, weight_path):
        """
        Load the Mask R-CNN model and its weights.
        """
        model_conf = {
            "model_name": "Mask_RCNN_Resnet_50_FPN",
            "pre_trained": False,
            "loaded": False,
            "load_source": "outputs/All_RGB_SSL/rotnet_pt_1/rotnet_best.pth",
            "trainable_layers": 5,
            "num_classes": 2
            }
        model = ModelBuilder(model_conf).model_builder()
        state_dict = torch.load(weight_path)
        state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        model.to("cuda:0")
        return model
    
    def get_mask(self, image, mask, colour, alpha = 0.5):
        """
        Perform inference on each image in the image directory, and add the results
        to the COCO annotations.
        """
        # Create a copy of the input image
        blended_image = image.copy()

        # Apply the mask and blend it with the original image using the alpha value
        for c in range(3):
            blended_image[:, :, c] = np.where(mask > 0.5, image[:, :, c] * (1 - alpha) + colour[c] * alpha, image[:, :, c])

        return blended_image
    
    def mask_to_polygons(self, mask):
        """
        Convert binary mask to polygons.
        """
        # Find contours
        mask = np.where(mask > 0, 1, mask).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Calculate the min and max non-zero coordinates for y and x
        coords = np.argwhere(mask)
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        print(x_max, y_max, x_min, y_min)

        nmask = mask[x_min:x_max+1, y_min:y_max+1]

        transitions = []
        for i, row in enumerate(nmask):
            prev_value = 0
            for j, value in enumerate(row):
                print(value)
                if prev_value != value:
                    # Record transition from 0 to 1 or 1 to 0
                    transitions.append((i + x_min, j + y_min))
                    prev_value = value

        print(transitions)

        # Convert contours to polygons
        polygons = []
        for contour in contours:
            # Approximate the contour to a polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Flatten the polygon array and append to polygons list
            polygons.append(approx.flatten().tolist())

        return polygons
    
    def get_predictions(self):
        """
        Perform inference on each image in the image directory, and add the results
        to the COCO annotations.
        """

        # define image tranfrom
        transfom = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        count = 1
        for img_file in os.listdir(self.img_dir):
            base_name, _ = os.path.splitext(img_file)  # separate the file name from the extension

            # image path
            image_path = os.path.join(self.img_dir, img_file)

            # load image
            image = Image.open(image_path)
            np_img = np.array(image)

            # image tensor
            img_tensor = transfom(image)
            img_tensor = img_tensor.cuda()  # assign the GPU tensor back to img_tensor

            # get model output
            with torch.no_grad():
                output = self.model([img_tensor])

            # Move tensors to CPU and convert to numpy arrays
            output = [{k: v.detach().cpu().numpy() for k, v in t.items()} for t in output]

            # Explicitly delete the img_tensor variable to save GPU memory
            del img_tensor
            torch.cuda.empty_cache()
            
            # outputs
            masks = output[0]['masks']
            scores = output[0]['scores']
            bbox = output[0]['boxes']
    
            # apply masks from instances
            for i in range(masks.shape[0]):
                if scores[i] > 0.5:
                    mask = masks[i][0]
                    colour = np.random.randint(0, 256, 3)
                    np_img = self.get_mask(np_img, mask, colour)

                    polygons = self.mask_to_polygons(mask)
        
                    x1, y1, x2, y2 = bbox[i].tolist()
                    width = x2 - x1
                    height = y2 - y1

                    self.coco_annotations["annotations"].append({
                        "image_id": len(self.coco_annotations["images"]),
                        "bbox": [x1, y1, width, height],
                        "category_id": 1,
                        "id": len(self.coco_annotations["annotations"]),
                        "segmentation": [polygons],  # This is not COCO format, but it's the best we can do without more complex processing
                        })

            # Save and clear image data to conserve memory
            title = self.out_dir + "/img_" + str(count) + ".jpg"
            img =  cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(title, img)
            np_img = None  # Added this to free memory
            
            
            #####################################################################

            # Add image information to the annotations
            self.coco_annotations["images"].append({
                "id": len(self.coco_annotations["images"]),
                "file_name": img_file,
                "height": image.height,
                "width": image.width
                })

            # Save annotations after each image
            self.save_json(base_name)

            # Clear the annotations for the next image
            self.coco_annotations["annotations"].clear()
            
            print(f"labelled: {count}")
            count += 1
            # Collect garbage
            gc.collect()

            # Pause for 1 second
            #time.sleep(0.5)
    
    def save_json(self, file_name):
        """
        Save the COCO annotations to a JSON file in the output directory.
        """
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        with open(os.path.join(self.out_dir, f"{file_name}_coco_annotations.json"), "w") as f:
            json.dump(self.coco_annotations, f)

    def generate(self):
        """
        Generate the COCO annotations for the images in the image directory.
        """
        self.get_predictions()

# execution
if __name__ == "__main__":
    
    weight_path = "outputs/All_RGB_SSL/Rotnet_pt_Mask_RCNN_5/best.pth"
    img_dir = "datasets/sources/All_RGB"
    out_dir = "tools/label_generator_ouputs/"
    
    ps_label = PseudoLabeller(weight_path, img_dir, out_dir)
    ps_label.generate()