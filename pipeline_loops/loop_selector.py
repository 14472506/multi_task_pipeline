"""
Details
"""
# impori
from .instance_loops import training_loop, val_loop, test_loop

# class
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
            return training_loop

    def get_val(self):

        if self.cfg["loop_type"] == "instance":
            return val_loop

    def get_test(self):

        if self.cfg["loop_type"] == "instance": 
            return test_loop

    



