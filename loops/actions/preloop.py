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
            "jigsaw": self._classifier_action,
            "mask_rcnn": self._instance_seg_action,
            "rotmask_multi_task": self._multitask_action,
            "dual_mask_multi_task": self._multitask_action
        }
        return self.action_map[self.model_name]
    
    def _classifier_action(self, model, cfg):
        """ Detials """
        banner = "================================================================================"
        title = " Classifier Training "

        print(banner)
        print(title)
        print(banner)

    def _instance_seg_action(self, model, cfg):
        """ Details """
        banner = "================================================================================"
        title = " Instance Seg Training "

        # Load model weights here.

        print(banner)
        print(title)
        print(banner)
    
    def _multitask_action(self, model, cfg):
        """ Detials """
        banner = "================================================================================"
        title = " Multi Task Training "

        print(banner)
        print(title)
        print(banner)