# imports
from pycocotools.coco import COCO
from pycocotools.mask import encode, decode, area
from PIL import Image
import numpy as np
import json
import torch
import torch.nn.functional as F
from skimage import measure


class ImageFrometter():
    """ Detials """
    def __init__(self, path, img_root, crop_h, crop_w, rs_h, rs_w):
        """ Detials """
        self.path = path
        self.img_root = img_root
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.rs_h = rs_h
        self.rs_w = rs_w
        self.idx_count = 0
        
        self._initialise_params()

    def _initialise_params(self):
        """ Detials """
        self.coco = COCO(self.path)
        
    def main(self):
        """ Detials """     
        for img_id in self.coco.imgs:
            img_info = self.coco.loadImgs(img_id)[0]
            width, height = img_info['width'], img_info['height']

            if width > self.crop_w:
                
                left = int((width - self.crop_w) / 2)
                top = int((height - self.crop_h) / 2)
                right = int((width + self.crop_w) / 2)
                bottom = int((height + self.crop_h) / 2)

                # process annotations
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                anns = self.coco.loadAnns(ann_ids)
                for ann in anns:
                    mask = self.coco.annToMask(ann)
                    mask == ann["category_id"]
                    cropped_mask = mask[top:bottom, left:right]
                    resized_mask = Image.fromarray(cropped_mask).resize((self.rs_w, self.rs_h), Image.NEAREST)
                    resized_mask = np.array(resized_mask).astype(np.uint8)
                    # area 
                    fortran_array = np.asfortranarray(resized_mask)
                    encoded_mask = encode(fortran_array)
                    rs_area = int(area(encoded_mask))
                    contours =  measure.find_contours(resized_mask) 
                    proposals = []
                    contours = measure.find_contours(resized_mask)
                    annotations = []
                    for contour in contours:
                        contour = np.flip(contour, axis=1)
                        segmentation = contour.ravel().tolist(),
                        proposals.append(segmentation[0])
                    #print("### Props ###")
                    #print(len(proposals))
                    longest_proposal = max(proposals, key=len)
                    annotations.append(longest_proposal)
                    # get brounding box 
                    rows = np.any(resized_mask, axis=1)
                    cols = np.any(resized_mask, axis=0)
                    ymin, ymax = np.where(rows)[0][[0, -1]]
                    xmin, xmax = np.where(cols)[0][[0, -1]]
                    ann["bbox"][0] = int(xmin)
                    ann["bbox"][1] = int(ymin)
                    ann["bbox"][2] = int(xmax - xmin)
                    ann["bbox"][3] = int(ymax - ymin)
                    ann["area"] = rs_area
                    ann["segmentation"] = annotations   
                #image_path = self.img_root + img_info["file_name"]
                #image = Image.open(image_path)
                #cropped_img = image.crop((left, top, right, bottom))
                #resized_img = cropped_img.resize((1920, 1080))
                #resized_img.save(img_info["file_name"])

        for dct in self.coco.dataset["images"]:
            if dct["width"] != 1920:
                dct["width"] = 1920
            if dct["height"] != 1080:
                dct["height"] = 1080 

        # Save modified annotations
        with open('val.json', 'w') as f:
            json.dump(self.coco.dataset, f)
                
if __name__ == "__main__":
    F = ImageFrometter(
        "data_handling/sources/jersey_dataset_v4/val/val.json",
        "data_handling/sources/jersey_dataset_v4/val/",
        1836,
        3264,
        1080,
        1920
    )
    F.main()

                