"""
Detials
"""
# imports
import torch
from torchmetrics.detection import MeanAveragePrecision
import pickle
import os
import numpy as np
from .coco_eval_files.coco_evaluation import evaluate
import gc

# class
class TestAction():
    """ Detials """
    def __init__(self, cfg):
        """ Detials """
        self.cfg = cfg
        self.model_name = self.cfg["model_name"]
    
    def action(self):
        self.action_map = {
            "rotnet_resnet_50": self._classifier_action,
            "jigsaw": self._classifier_action,
            "mask_rcnn": self._instance_seg_action,
            "rotmask_multi_task": self._multitask_action
        }
        return self.action_map[self.model_name]
    
    def _classifier_action(self):
        """ Detials """
        print("testing")

    def _instance_seg_action(self, model, loader, step, logger, device):
        """ Detials """
        def mAP_eval(model, loader, logger, device, eval_type="segm", model_type="pre"):
            """ Details """
            # init eval
            metric = MeanAveragePrecision(iou_type = eval_type)
            logger.load_model(model, model_type)
            model.eval()

            for i, data in enumerate(loader):
                input, target = data
                input = list(image.to(device) for image in input)
        
                with torch.autocast("cuda"):
                    with torch.no_grad():
                        predictions = model(input)

                masks_in = predictions[0]["masks"].detach().cpu()
                masks_in = masks_in > 0.5
                masks_in = masks_in.squeeze(1) 
                targs_masks = target[0]["masks"].bool()
                targs_masks = targs_masks.squeeze(1)  
                preds = [dict(masks=masks_in, scores=predictions[0]["scores"].detach().cpu(), labels=predictions[0]["labels"].detach().cpu(),)]
                targs = [dict(masks=targs_masks, labels=target[0]["labels"],)]
                metric.update(preds, targs)

                del predictions, input, target, masks_in, targs_masks, preds, targs
                torch.cuda.empty_cache()
                print("batch %s complete" %(i))
            
            res = metric.compute()
            return res

        results = {}
        pre_mAP = mAP_eval(model, loader, logger, device, eval_type="segm", model_type="pre")
        pre_mAP = self._convert_to_dict(pre_mAP)
        print(pre_mAP)
        results["pre_step"] = pre_mAP
        if step:
            post_mAP = mAP_eval(model, loader, logger, device, eval_type="segm", model_type="post")
            post_mAP = self._convert_to_dict(post_mAP)
            results["post_step"] = post_mAP
        logger.save_results(results)

    def _convert_to_dict(self, dict):
        """ Detials """
        for key, value in dict.items():
            if isinstance(value, torch.Tensor):
                dict[key] = value.item()
        return(dict)
    
    def _multitask_action(self, model, loader, step, logger, device):
        """ Detials """
        def mAP_eval(model, loader, logger, device, eval_type="segm", model_type="pre"):
            """ Details """
            # init eval
            metric = MeanAveragePrecision(iou_type = eval_type)
            logger.load_model(model, model_type)
            model.eval()

            for i, data in enumerate(loader):
                input, target, _, _ = data
                input = list(image.to(device) for image in input)
        
                with torch.autocast("cuda"):
                    with torch.no_grad():
                        predictions = model(input, mode="segm")

                masks_in = predictions[0]["masks"].cpu().detach()
                masks_in = masks_in > 0.5
                masks_in = masks_in.squeeze(1) 
                targs_masks = target[0]["masks"].bool()
                targs_masks = targs_masks.squeeze(1)  
                preds = [dict(masks=masks_in, scores=predictions[0]["scores"].cpu().detach(), labels=predictions[0]["labels"].cpu().detach(),)]
                targs = [dict(masks=targs_masks, labels=target[0]["labels"],)]
                metric.update(preds, targs)

                del predictions, input, target
                torch.cuda.empty_cache()
                print("batch %s complete" %(i))
            
            res = metric.compute()
            return res
        
        results = {}
        pre_mAP = mAP_eval(model, loader[0], logger, device, eval_type="segm", model_type="pre")
        pre_mAP = self._convert_to_dict(pre_mAP)
        print(pre_mAP)
        results["pre_step"] = pre_mAP
        if step:
            post_mAP = mAP_eval(model, loader[0], logger, device, eval_type="segm", model_type="post")
            post_mAP = self._convert_to_dict(post_mAP)
            results["post_step"] = post_mAP
        logger.save_results(results)

    ################################

    def _multitask_action_coco(self, model, loader, step, logger, device):
        """ Detials """
        results = {}
        loader = loader[0]
        
        logger.load_model(model, "pre")
        pre_mAP = evaluate(model, loader, device, "_", train_flag=True)
        #pre_mAP = self._convert_to_dict(pre_mAP)
        print(pre_mAP)
        results["pre_step"] = pre_mAP
        if step:
            logger.load_model(model, "post")
            post_mAP = evaluate(model, loader, device, "_", train_flag=True)
            #post_mAP = self._convert_to_dict(post_mAP)
            results["post_step"] = post_mAP
        logger.save_results(results)






