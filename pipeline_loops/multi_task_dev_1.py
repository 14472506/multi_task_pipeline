"""
Detials
"""
# import
import torch
import gc
from pipeline_logging import train_reporter_multi_task, val_reporter_multi_task
from utils.coco_evaluation import evaluate
from utils import garbage_collector
import torch.nn.functional as F

# training loop
def multi_task_training_loop(model, loader, optimizer, scaler, logger, device, iter_count, epoch, awl=None):
    """ implementing multi class training loop """
    model.train()
    
    # init loader detials
    joint_loader, ssl_loader = loader
    joint_iter = iter(joint_loader)
    ssl_iter = iter(ssl_loader)

    loop_loss_acc = 0

    # adjusting larger loader to correct num of instances
    num_extra_fetches = iter_count % len(ssl_loader)
    if num_extra_fetches:
        print(num_extra_fetches)
        for i in range(num_extra_fetches):
            ssl_img, ssl_target = next(ssl_iter) 

    accumulation_iter = 2
        
    # loop over labelled dataset
    for i in range(len(joint_iter)):
        
        # collect iteration data
        joint_img, joint_masks, joint_ssl_img, joint_ssl_target = next(joint_iter)
        joint_img = list(image.to(device) for image in joint_img)
        joint_masks = [{k: v.to(device) for k, v in t.items()} for t in joint_masks]
        joint_ssl_img = joint_ssl_img[0].to(device)
        joint_ssl_target = joint_ssl_target[0].to(device)

        # this seems lazy as fuck but if it works it works
        try:
            ssl_img, ssl_target = next(ssl_iter) 
        except StopIteration:
            print("resetting iter")
            ssl_iter = iter(ssl_loader)
            ssl_img, ssl_target = next(ssl_iter)         
        ssl_img = ssl_img.to(device)
        ssl_target = ssl_target.to(device)

        # get model output
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            joint_sup_loss_dict = model(joint_img, joint_masks, task="supervised")
            joint_sup_losses = sum(loss for loss in joint_sup_loss_dict.values())
            joint_ssl_pred = model(joint_ssl_img, task="self_supervised")
            joint_ssl_loss = F.cross_entropy(joint_ssl_pred[0], joint_ssl_target)
            ssl_pred = model(ssl_img, task="self_supervised")
            ssl_loss = F.cross_entropy(ssl_pred, ssl_target)

            #automatic weighted loss
            weighted_losses = awl(joint_sup_losses, joint_ssl_loss, ssl_loss)
        
        # scaler step analogous to optimiser
        scaler.scale(weighted_losses).backward()

        if (i+1) % accumulation_iter == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
      
        # reporting
        loop_loss_acc += weighted_losses.item()
        train_reporter_multi_task(iter_count, device, weighted_losses.item(), joint_sup_losses.item(), joint_ssl_loss.item(), ssl_loss.item(), epoch, "multi_task")
       
        iter_count += 1
        garbage_collector()

    logger["train_loss"].append(loop_loss_acc/len(joint_loader)) 
    return iter_count

def multi_task_validation_loop(model, loader, scaler, logger, device, epoch, path, awl):
    """ Detials """
    model.train()
    
    # init loader detials
    joint_loader, ssl_loader = loader
    joint_iter = iter(joint_loader)
    ssl_iter = iter(ssl_loader)

    awl_acc = 0
    joint_mask_acc = 0
    joint_ssl_acc = 0
    ssl_acc = 0

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
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                joint_sup_loss_dict = model(joint_img, joint_masks, task="supervised")
                joint_sup_losses = sum(loss for loss in joint_sup_loss_dict.values())
                joint_ssl_pred = model(joint_ssl_img, task="self_supervised")
                joint_ssl_loss = F.cross_entropy(joint_ssl_pred[0], joint_ssl_target)
                ssl_pred = model(ssl_img, task="self_supervised")
                ssl_loss = F.cross_entropy(ssl_pred, ssl_target)

                #automatic weighted loss
                weighted_losses = awl(joint_sup_losses, joint_ssl_loss, ssl_loss)
      
        # reporting
        awl_acc += weighted_losses.item()
        joint_mask_acc += joint_sup_losses.item()
        joint_ssl_acc += joint_ssl_loss.item()
        ssl_acc += ssl_loss.item()
        garbage_collector()

    awl_loss = awl_acc/len(joint_loader)
    joint_mask_loss = joint_mask_acc/len(joint_loader)
    joint_self_loss = joint_ssl_acc/len(joint_loader)
    self_loss = ssl_acc/len(joint_loader)
    logger["val_loss"].append(joint_mask_loss)
    val_reporter_multi_task(device, awl_loss, joint_mask_loss, joint_self_loss, self_loss, "multi_task", epoch)
    

def multi_task_testing_loop(model, loader, device, exp_dir, train_flag=True):
    """ Detials """
    # extract map loader 
    loader = loader[0]

    # extract Mask R-CNN model
    model.train()

    mAP = evaluate(model, loader, device, exp_dir, train_flag)
    print(mAP)
    garbage_collector()