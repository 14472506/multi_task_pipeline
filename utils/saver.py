"""
Detials
"""
# imports
import os
import errno
import torch
import json

# functions
def make_dir(path):
    """
    Detials
    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def best_loss_saver(epoch, logger, model, optimizer, path, pth_name):
    """
    Detials
    """
    # if model is first model
    if logger["epochs"][-1] == 0:
        save_model(epoch, model, optimizer, path, pth_name)
        logger["best_val"].append(logger["val_loss"][-1])
        logger["best_epoch"].append(logger["epochs"][-1])

    if logger["val_loss"][-1] < logger["best_val"][-1]:
        save_model(epoch, model, optimizer, path, pth_name)
        logger["best_val"].append(logger["val_loss"][-1])
        logger["best_epoch"].append(logger["epochs"][-1])

def save_model(epoch, model, optimizer, path, pth_name):

    make_dir(path) # check or make dir
    model_path = path + "/" + pth_name

    checkpoint = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, model_path)

def save_json(file, path, json_name):
    # saving data in json
    make_dir(path) # check or make dir
    save_file = path + "/" + json_name
    with open(save_file, "w") as f:
        json.dump(file, f)