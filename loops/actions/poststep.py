"""
Detials
"""
# imports

# class
class PostStep():
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
        pass