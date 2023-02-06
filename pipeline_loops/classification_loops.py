"""
Details
"""
# imports
import torch
from utils import classification_loss as criterion
from pipeline_logging import train_reporter, val_reporter

# functions 
def class_train_loop(model, loader, optimizer, logger, device ,iter_count, epoch):
    """
    Detials
    """
    # model config
    model.train()

    # init loop counters
    loop_loss_acc = 0

    # accume init here
    for i, data in enumerate(loader):

        # extact data 
        image, target = data
        image, target = image.to(device), target.to(device)

        # forward
        pred = model(image)
        loss = criterion(pred, target)

        # backward + optimizer_step
        optimizer.zero_grad()
        loss.backward()        
        optimizer.step()
        
        # loss detach ------ May not be needed
        loss = loss.detach()
        loop_loss_acc += loss.item()
        train_reporter(iter_count, device, loss.item(), epoch)
        iter_count += 1
    
    logger["train_loss"].append(loop_loss_acc/len(loader))

    return iter_count

def class_val_loop(model, loader, logger, device, epoch):
    """
    Detials
    """
    # model config
    model.eval()

    # init loop counters
    loop_loss_acc = 0

    # accume init here
    for i, data in enumerate(loader):

        # extact data 
        image, target = data
        image, target = image.to(device), target.to(device)

        # forwards 
        with torch.no_grad():
            pred = model(image)
        
        # get loss
        loss = criterion(pred, target)

        # accumulate loss
        loop_loss_acc += loss.item()
    
    # get val loss for epoch
    loss = loop_loss_acc/len(loader)
    logger["val_loss"].append(loss)
    val_reporter(device, loss, epoch)

def class_test_loop(model, test_loader, device, exp_dir, train_flag=True):
    print("TEST LOOP")
