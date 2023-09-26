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
            "rotnet_resnet_50": self._classifier_action,
            "jigsaw": self._classifier_action,
            "mask_rcnn": self._instance_seg_action,
            "rotmask_multi_task": self._multitask_action
        }
        return self.action_map[self.model_name]

    def _classifier_action(self, epoch):
        """ Detials """
        banner = "--------------------------------------------------------------------------------"
        title = "Epoch: " + str(epoch)

        print(banner)
        print(title)
        print(banner)

    def _instance_seg_action(self, epoch):
        """ Detials """
        banner = "--------------------------------------------------------------------------------"
        title = "Epoch: " + str(epoch)

        print(banner)
        print(title)
        print(banner)
    
    def _multitask_action(self, epoch):
        """ Detials """
        banner = "--------------------------------------------------------------------------------"
        title = "Epoch: " + str(epoch)

        print(banner)
        print(title)
        print(banner)
