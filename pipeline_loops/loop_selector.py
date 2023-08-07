"""
Details
"""
# imports
from .instance_loops import (inst_train_loop, 
                             inst_val_loop,
                             inst_test_loop)
from .classification_loops import(class_train_loop,
                                  class_val_loop,
                                  class_test_loop)
from .multi_task_dev_3 import(multi_train_loop,
                             multi_val_loop,
                             multi_test_loop)

class LoopSelector():
    """
    Details
    """
    def __init__(self, cfg):
        """
        Detials
        """
        self.cfg = cfg

    def get_train(self):

        if self.cfg["loop_type"] == "instance":
            return inst_train_loop
        if self.cfg["loop_type"] == "classification":
            return class_train_loop
        if self.cfg["loop_type"] == "multi_task":
            return multi_train_loop

    def get_val(self):

        if self.cfg["loop_type"] == "instance":
            return inst_val_loop
        if self.cfg["loop_type"] == "classification":
            return class_val_loop
        if self.cfg["loop_type"] == "multi_task":
            return multi_val_loop

    def get_test(self):

        if self.cfg["loop_type"] == "instance": 
            return inst_test_loop
        if self.cfg["loop_type"] == "multi_task":
            return multi_test_loop

    



