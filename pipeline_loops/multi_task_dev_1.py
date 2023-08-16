"""
Detials
"""
# import
import torch
import gc
from pipeline_logging import train_reporter, val_reporter
from utils import classification_loss as criterion
from utils.coco_evaluation import evaluate
from utils import garbage_collector

# training loop
def multi_task_training_loop(model, loader, optimizer, scaler, logger, device, iter_count, epoch):
    """ implementing multi class training loop """

    # set model to train
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
        joint_ssl_img = list(image.to(device) for image in joint_ssl_img)
        joint_ssl_target = list(label.to(device) for label in joint_ssl_target)

        ssl_img, ssl_target = next(ssl_iter) 
        ssl_img = list(image.to(device) for image in ssl_img)
        ssl_target = list(label.to(device) for label in ssl_target)

        # get model output
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            joint_sup_loss_dict = model(joint_img, joint_masks)
            joint_sup_losses = sum(loss for loss in joint_sup_loss_dict.values())
            joint_ssl_pred = model(joint_ssl_img)
            joint_ssl_loss = criterion(joint_ssl_pred, joint_ssl_target)
            ssl_pred = model(ssl_img)
            ssl_loss = criterion(ssl_pred, ssl_target)
        
        # weighted losses
        # weighted_losses = AWL(joint_sup_losses, joint_ssl_loss, ssl_loss)

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













# basic loops
def multi_train_loop(model, loader, optimizer, scaler, logger, device ,iter_count, epoch):
    """
    Details
    """
    # set model to train
    model.train()

    # init loop counters
    ssl_loop_loss_acc = 0
    loop_loss_acc = 0

    downstream_loader = loader[0]
    upstream_loader = loader[1]

#    for i, upstream_data in enumerate(upstream_loader):
#
#        # extact data 
#        upstream_image, upstream_target = upstream_data
#        upstream_image, upstream_target = upstream_image.to(device), upstream_target.to(device)
#        
#        # forward
#        upstream_pred = model("ss", upstream_image)
#        upstream_loss = criterion(upstream_pred, upstream_target)
#
#        # backward + optimizer_step
#        optimizer.zero_grad()
#        upstream_loss.backward()        
#        optimizer.step()
#        
#        # loss detach ------ May not be needed
#        upstream_loss = upstream_loss.detach()
#        ssl_loop_loss_acc += upstream_loss.item()
#        train_reporter(iter_count, device, upstream_loss.item(), epoch)
#        iter_count += 1

    # enter loader loop
    for j, downsteam_data in enumerate(downstream_loader):
        
        # load data for loader
        downstream_images, downstream_targets, downstream_rot, downstream_rot_res = downsteam_data
        downstream_images = list(downstream_image.to(device) for downstream_image in downstream_images)
        downstream_targets = [{k: v.to(device) for k, v in t.items()} for t in downstream_targets]


        print(downstream_rot)
        print(downstream_rot_res)

        # get model output
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            downsteam_loss_dict = model("mask", downstream_images, downstream_targets)
            downsteam_losses = sum(loss for loss in downsteam_loss_dict.values())

        scaler.scale(downsteam_losses).backward()
        #
        #scaler.unscale_(optimizer)
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()
        
        #losses.backward()
        #optimizer.step()
        optimizer.zero_grad()
        
        # reporting
        loop_loss_acc += downsteam_losses.item()
        train_reporter(iter_count, device, downsteam_losses.item(), epoch)
        iter_count += 1

        garbage_collector()

    logger["ssl_train_loss"].append(ssl_loop_loss_acc/len(upstream_loader))
    logger["train_loss"].append(loop_loss_acc/len(downstream_loader))
    
    return iter_count

def multi_val_loop(model, loader, scaler, logger, device, epoch, exp_dir):
    """
    Details
    """
    # set model to train
    model.train()

    # init loop counters
    ssl_loop_loss_acc = 0
    loop_loss_acc = 0

    downstream_loader = loader[0]
    upstream_loader = loader[1]

    for i, upstream_data in enumerate(upstream_loader):

        # extact data 
        upstream_image, upstream_target = upstream_data
        upstream_image, upstream_target = upstream_image.to(device), upstream_target.to(device)

        # forward
        with torch.no_grad():
            upstream_pred = model("ss", upstream_image)
            upstream_loss = criterion(upstream_pred, upstream_target)

        ssl_loop_loss_acc += upstream_loss.item()

    # enter loader loop
    for j, downsteam_data in enumerate(downstream_loader):
        
        # load data for loader
        downsteam_images, downsteam_targets = downsteam_data
        downsteam_images = list(downsteam_image.to(device) for downsteam_image in downsteam_images)
        downsteam_targets = [{k: v.to(device) for k, v in t.items()} for t in downsteam_targets]

        # get model output
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                downsteam_loss_dict = model("mask", downsteam_images, downsteam_targets)
                downsteam_losses = sum(loss for loss in downsteam_loss_dict.values())

        loop_loss_acc += downsteam_losses.item()

        garbage_collector()

    #mAP = evaluate(model,
    #    loader, 
    #    device,
    #    exp_dir, 
    #    train_flag=True)
    #print(mAP)

    loss = loop_loss_acc/len(downstream_loader)
    ssl_loss = ssl_loop_loss_acc/len(upstream_loader)
    logger["val_loss"].append(loss)
    logger["ssl_val_loss"].append(ssl_loss)
    print("SSL Loss: ", ssl_loss)
    val_reporter(device, loss, epoch)
    
def multi_test_loop(model, loader, device, exp_dir, train_flag=True):
    garbage_collector()
    test_loader = loader[0]
    mAP = evaluate(model, test_loader, device, exp_dir, train_flag)
    print(mAP)