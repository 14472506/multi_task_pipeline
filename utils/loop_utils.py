"""
Details
"""
# imports
import torch

# functions
def schedule_loader(model, path, pth_name):
    """
    Detials
    """
    # load best model so far
    load_path = path + "/" + pth_name
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint["state_dict"])

    # save best model as best pre step model
    save_path = path + "/ps_" + pth_name
    torch.save(checkpoint, save_path)

def load_model(path, pth_name, model):
    """
    Detials
    """
    # load best model so far
    load_path = path + "/" + pth_name
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint["state_dict"])