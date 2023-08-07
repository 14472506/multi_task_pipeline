"""
Details
"""
# imports
import torch
import gc
#PLACEHOLDER FOR: from pipeline_logging import train_reporter, val_reporter
from utils import classification_loss as criterion
from utils.coco_evaluation import evaluate
from utils import garbage_collector, multi_task_training_scheduler

# class
def multi_train_loop(model, loaders, optimizer, scaler, logger, device, iter_count, epoch):
    """
    Detials
    """
    
    # loop settings
    model.train()
    
    # data loader unpacking
    supervised_loader = loaders[0]
    self_supervised_loader = loaders[1]

    # initialising data loader itteration
    supervised_iterator = iter(supervised_loader)
    self_supervised_iterator = iter(self_supervised_loader)

def multi_val_loop(something):
    pass

def multi_test_loop(something):
    pass



