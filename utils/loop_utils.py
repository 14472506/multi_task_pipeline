"""
Details
"""
# imports
import torch
import torch.nn.functional as F
import gc

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

# loss functions
def classification_loss(y_hat, y):
    """
    Detials
    """
    loss = F.cross_entropy(y_hat, y)
    return loss

def garbage_collector():
    """
    Details
    """
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()

def multi_task_training_scheduler(max_list, min_list):
    """
    something
    """
    if len(min_list) == 0:
        return(max_list)
    else:
        even = int(len(max_list)/len(min_list))
        rem = len(max_list)%len(min_list)
        iters = len(max_list) - rem
        new_max = []
        min_count = 0
        for i in range(iters):
            if i%even == 0:
                new_max.append(min_list[min_count])
                min_count += 1
            new_max.append(max_list[i])
        if rem == 0:
            new_min = []
        else:
            new_min = max_list[-rem:]
        return multi_task_training_scheduler(new_max, new_min)