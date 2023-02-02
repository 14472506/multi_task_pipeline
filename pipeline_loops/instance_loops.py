"""
Detials
"""
# import
import torch
import gc

# basic loops
def inst_train_loop(model, loader, optimizer, device):
    """
    Details
    """
    # set model to train
    model.train()

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

        print(losses.item())


        



def inst_val_loop(a, b):
    print(a+b)

def inst_test_loop(a, b):
    print(a+b)