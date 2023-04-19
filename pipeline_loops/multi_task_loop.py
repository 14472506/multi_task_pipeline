"""
Detials
"""
# import
import torch
import gc
from pipeline_logging import train_reporter, val_reporter
from utils import classification_loss as criterion
from utils.coco_evaluation import evaluate
from utils import garbage_collector, multi_task_training_scheduler

# basic loops
def multi_train_loop(model, loaders, optimizer, scaler, logger, device ,iter_count, epoch):
    """
    Details
    """
    # loop setup
    model.train()                               # set model to train
    loader = loaders[0]                         # get supervised loader
    ssl_loader = loaders[1]                     # get self supervised loader
    iterator = iter(loader)                     # get supervised dataloader iterator
    ssl_iterator = iter(ssl_loader)             # get self supervised dataloader iteratior
    acc_loss = 0                                # accumulated supervised loss
    ssl_acc_loss = 0                            # accumulaled self supervised loss

    # IMPORTANT. SOME LOGIC TO SELECT LARGEST LIST IS NEEDED HERE
    sup_list = ["SL"]*len(loader)
    ssl_list = ["SSL"]*len(ssl_loader)
    program = multi_task_training_scheduler(ssl_list, sup_list)

    # enter epoch loop
    for i, flag in enumerate(program):
        if flag == "SL":
            # ===== Instance and ssl ===== #
            # extract data from current loader and sending it to davice
            inst_img, inst_targ, rot_img, rot_targ = next(iterator)
            inst_img = list(im.to(device) for im in inst_img)
            inst_targ = [{k: v.to(device) for k, v in t.items()} for t in inst_targ]
            rot_img = torch.stack(list(im.to(device) for im in rot_img))
            rot_targ = torch.stack(list(im.to(device) for im in rot_targ))

            # carrying out forward pass
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                ssl_pred, loss_dict = model(rot_img, inst_img, inst_targ)
                ssl_loss = criterion(ssl_pred[0], rot_targ[0])
                losses = sum(loss for loss in loss_dict.values())
                total_loss = ssl_loss + losses

            acc_loss += total_loss.item() 

        if flag == "SSL":
            img, targ = next(ssl_iterator)
            img, targ = img.to(device), targ.to(device)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                pred = model(img)
                total_loss = criterion(pred, targ)

            ssl_acc_loss += total_loss.item()

        # backwards and optimizer step
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # setting optimizer to zero grad
        optimizer.zero_grad()

        # Test Reporting
        train_reporter(iter_count, device, total_loss.item(), epoch, "combined_loss")       
        iter_count += 1

        garbage_collector()

    # supervised loss
    loss = acc_loss/len(loader)
    logger["train_loss"].append(loss)
    ssl_loss = ssl_acc_loss/len(ssl_loader)
    logger["ssl_train_loss"].append(ssl_loss)
    
    return iter_count

def multi_val_loop(model, loaders, scaler, logger, device, epoch, exp_dir):
    """
    Details
    """
    # loop setup
    model.train()                       # set model to train
    loader = loaders[0]                 # get supervised loader
    ssl_loader = loaders[1]             # get self supervised loader
    epoch_len = len(loader)#len(ssl_loader) + len(loader) # get epoch length
    iterator = iter(loader)             # get supervised dataloader iterator
    ssl_iterator = iter(ssl_loader)     # get self supervised dataloader iteratior
    acc_loss = 0                        # accumulated supervised loss
    ssl_acc_loss = 0                    # accumulaled self supervised loss

    # IMPORTANT. SOME LOGIC TO SELECT LARGEST LIST IS NEEDED HERE
    sup_list = ["SL"]*len(loader)
    ssl_list = ["SSL"]*len(ssl_loader)
    program = multi_task_training_scheduler(ssl_list, sup_list)

    # enter epoch loop
    for i, flag in enumerate(program):
        if flag == "SL":
            # ===== Instance and ssl ===== #
            # extract data from current loader and sending it to davice
            inst_img, inst_targ, rot_img, rot_targ = next(iterator)
            inst_img = list(im.to(device) for im in inst_img)
            inst_targ = [{k: v.to(device) for k, v in t.items()} for t in inst_targ]
            rot_img = torch.stack(list(im.to(device) for im in rot_img))
            rot_targ = torch.stack(list(im.to(device) for im in rot_targ))

            # carrying out forward pass
            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    ssl_pred, loss_dict = model(rot_img, inst_img, inst_targ)
                    ssl_loss = criterion(ssl_pred[0], rot_targ[0])
                    losses = sum(loss for loss in loss_dict.values())
                    total_loss = ssl_loss + losses

            acc_loss += total_loss.item() 

        if flag == "SSL":
            img, targ = next(ssl_iterator)
            img, targ = img.to(device), targ.to(device)

            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    pred = model(img)
                    total_loss = criterion(pred, targ)

            ssl_acc_loss += total_loss.item()

        garbage_collector()

    mAP = evaluate(model.Mask_RCNN,
        loader, 
        device,
        exp_dir, 
        train_flag=True)

    loss = acc_loss/len(loader)
    ssl_loss = ssl_acc_loss/len(ssl_loader)
    logger["val_loss"].append(loss)
    logger["ssl_val_loss"].append(ssl_loss)
    logger["mAP"].append(mAP)
    val_reporter(device, loss, "sup_loss", epoch)
    val_reporter(device, ssl_loss, "ssl_loss", epoch)
    val_reporter(device, mAP, "mAP", epoch)
    
def multi_test_loop(model, loader, device, exp_dir, train_flag=True):
    garbage_collector()
    test_loader = loader[0]
    mAP = evaluate(model, test_loader, device, exp_dir, train_flag)
    print(mAP)