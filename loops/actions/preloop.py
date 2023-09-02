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
            "rotnet_resnet_50": self._classifier_action
        }
        return self.action_map[self.model_name]
    
    def _classifier_action(self):
        """ Detials """
        banner = "================================================================================"
        title = " Classifier Training "

        print(banner)
        print(title)
        print(banner)