"""
Details
"""
# imports
from .instance_loops import (
    inst_train_loop, 
    inst_val_loop,
    inst_test_loop
)
from .classification_loops import (
    class_train_loop,
    class_val_loop,
    class_test_loop
)
from .multi_task_dev_3 import (
    multi_train_loop,
    multi_val_loop,
    multi_test_loop
)

class LoopSelector():
    """
    Details
    """
    def __init__(self, cfg):
        """
        Details
        """
        self.cfg = cfg
        self.loop_mappings = {
            "instance": {
                "train": inst_train_loop,
                "val": inst_val_loop,
                "test": inst_test_loop
            },
            "classification": {
                "train": class_train_loop,
                "val": class_val_loop,
                "test": class_test_loop
            },
            "multi_task": {
                "train": multi_train_loop,
                "val": multi_val_loop,
                "test": multi_test_loop
            }
        }

    def get_loop(self, loop_type):
        """
        Get the required loop based on the loop_type (train, val, test) and self.cfg["loop_type"]
        """
        try:
            return self.loop_mappings[self.cfg["type"]][loop_type]
        except KeyError:
            raise ValueError(f"Invalid loop type {self.cfg['loop_type']} or {loop_type}")

    # Wrapper functions for readability
    def get_train(self):
        return self.get_loop("train")

    def get_val(self):
        return self.get_loop("val")

    def get_test(self):
        return self.get_loop("test")
















"""
Details

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

    def __init__(self, cfg):

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
"""
    



