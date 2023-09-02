"""
Detials
"""
# imports

# class
class PreStep():
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

    def _classifier_action(self, epoch):
        """ Detials """
        banner = "--------------------------------------------------------------------------------"
        title = "Epoch: " + str(epoch)

        print(banner)
        print(title)
        print(banner)