"""
Detials
"""
# imports

# class
class PreLoop():
    """ Detials """
    def __init__(self, cfg):
        """ Detials """
        self.cfg = cfg
        self.model_name = self.cfg["model_name"]
    
    def action(self):
        self.action_map = {
            "rotnet_resnet_50": self._classifier_action,
            "rotmask_multi_task": self._multitask_action
        }
        return self.action_map[self.model_name]
    
    def _classifier_action(self):
        """ Detials """
        banner = "================================================================================"
        title = " Classifier Training "

        print(banner)
        print(title)
        print(banner)
    
    def _multitask_action(self):
        """ Detials """
        banner = "================================================================================"
        title = " Multi Task Training "

        print(banner)
        print(title)
        print(banner)