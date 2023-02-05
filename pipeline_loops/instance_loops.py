"""
Detials
"""
# import
import torch
import gc
from pipeline_logging import train_reporter, val_reporter
from utils.coco_evaluation import evaluate

# basic loops
def inst_train_loop(model, loader, optimizer, logger, device ,iter_count, epoch):
    """
    Details
    """
    # set model to train
    model.train()

    # init loop counters
    loop_loss_acc = 0

    # enter loader loop
    for i, data in enumerate(loader):
        
        # load data for loader
        images, targets = data
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # get model output
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # reporting
        loop_loss_acc += losses.item()
        train_reporter(iter_count, device, losses.item(), epoch)
        iter_count += 1

    logger["train_loss"].append(loop_loss_acc/len(loader))
    
    return iter_count

def inst_val_loop(model, loader, logger, device, epoch):
    """
    Details
    """
    # set model to train
    model.train()

    # init loop counters
    loop_loss_acc = 0

    # enter loader loop
    with torch.no_grad():
        for i, data in enumerate(loader):

            # load data for loader
            images, targets = data
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # get model output
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            loop_loss_acc += losses.item()

    loss = loop_loss_acc/len(loader)
    logger["val_loss"].append(loss)
    val_reporter(device, loss, epoch)
    
def inst_test_loop(model, test_loader, device, exp_dir, train_flag=True):
    mAP = evaluate(model, test_loader, device, exp_dir, train_flag)
    print(mAP)