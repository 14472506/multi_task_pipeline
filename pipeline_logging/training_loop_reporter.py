"""
Detials
"""
# imports
import torch

# functions
def train_reporter(iter_count, device, loss, epoch, print_freq=20):
    """
    Detials
    """
    if iter_count % print_freq == 0: #self.print_freq-1:
        # get GPU memory usage
        mem_all = torch.cuda.memory_allocated(device) / 1024**3 
        mem_res = torch.cuda.memory_reserved(device) / 1024**3 
        mem = mem_res + mem_all
        mem = round(mem, 2)
        print("[epoch: %s][iter: %s][memory use: %sGB] total_loss: %s" %(epoch ,iter_count, mem, loss))

def val_reporter(device, loss, epoch):

    # get GPU memory usage
    mem_all = torch.cuda.memory_allocated(device) / 1024**3 
    mem_res = torch.cuda.memory_reserved(device) / 1024**3 
    mem = mem_res + mem_all
    mem = round(mem, 2)
    print("[epoch: %s][iter: ---][memory use: %sGB] total_loss: %s" %(epoch, mem, loss))