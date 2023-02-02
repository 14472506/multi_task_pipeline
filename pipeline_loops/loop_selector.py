"""
Details
"""
# imports
from .instance_loops import (inst_train_loop, 
                             inst_val_loop,
                             inst_test_loop)
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

    def get_val(self):

        if self.cfg["loop_type"] == "instance":
            return inst_val_loop

    def get_test(self):

        if self.cfg["loop_type"] == "instance": 
            return inst_test_loop

    



