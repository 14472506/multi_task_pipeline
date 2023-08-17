"""
Detials
"""
# import
import torch
import gc
from pipeline_logging import train_reporter, val_reporter
from utils.coco_evaluation import evaluate
from utils import garbage_collector
import torch.nn.functional as F

# training loop
def multi_task_training_loop(model, loader, optimizer, scaler, logger, device, iter_count, epoch):
    """ implementing multi class training loop """
    model.train()
    
    # init loader detials
    joint_loader, ssl_loader = loader
    joint_iter = iter(joint_loader)
    ssl_iter = iter(ssl_loader)

    # loop over labelled dataset
    for i in range(len(joint_iter)):
        
        # collect iteration data
        joint_img, joint_masks, joint_ssl_img, joint_ssl_target = next(joint_iter)
        joint_img = list(image.to(device) for image in joint_img)
        joint_masks = [{k: v.to(device) for k, v in t.items()} for t in joint_masks]
        joint_ssl_img = joint_ssl_img[0].to(device)
        joint_ssl_target = joint_ssl_target[0].to(device)

        ssl_img, ssl_target = next(ssl_iter) 
        ssl_img = ssl_img.to(device)
        ssl_target = ssl_target.to(device)

        # get model output
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            joint_sup_loss_dict = model(joint_img, joint_masks, task="supervised")
            joint_sup_losses = sum(loss for loss in joint_sup_loss_dict.values())
            print("mrcnn done")
            joint_ssl_pred = model(joint_ssl_img, task="self_supervised")
            joint_ssl_loss = F.cross_entropy(joint_ssl_pred[0], joint_ssl_target)
            print("joint_ssl_done")
            ssl_pred = model(ssl_img, task="self_supervised")
            ssl_loss = F.cross_entropy(ssl_pred, ssl_target)
            print(ssl_loss)
        
        # weighted losses
        # weighted_losses = AWL(joint_sup_losses, joint_ssl_loss, ssl_loss)

        print("a fucking miracle has occured")

        scaler.scale(weighted_losses).backward()

        scaler.step(optimizer)
        scaler.update()
        
        #losses.backward()
        #optimizer.step()
        optimizer.zero_grad()
        
        # reporting
        loop_loss_acc += losses.item()
        train_reporter(iter_count, device, losses.item(), epoch, "mrcnn")
        iter_count += 1

        garbage_collector()

    logger["train_loss"].append(loop_loss_acc/len(loader)) 

def multi_task_validation_loop(model, loader, optimizer, scaler, logger, device, iter_count, epoch):
    """ Detials """
    pass

def multi_task_testing_loop(model, loader, optimizer, scaler, logger, device, iter_count, epoch):
    """ Detials """
    pass