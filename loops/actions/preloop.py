"""
Detials
"""
# imports
import torch

# class
class PreLoop():
    """ Detials """
    def __init__(self, cfg):
        """ Detials """
        self.cfg = cfg
        self.model_name = self.cfg["model_name"]
        self.load_model = self.cfg["load_model"]
        if self.load_model:
            self.load_path = self.cfg["params"]["load_path"]
    
    def action(self):
        self.action_map = {
            "rotnet_resnet_50": self._classifier_action,
            "jigsaw": self._classifier_action,
            "mask_rcnn": self._instance_seg_action,
            "rotmask_multi_task": self._multitask_action,
            "jigmask_multi_task": self._multitask_action,
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
    
    def _multitask_action(self, model, optimiser):
        """ Detials """

        if self.load_model:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(self.load_path)
            model.load_state_dict(checkpoint["state_dict"])
            optimiser.load_state_dict(checkpoint['optimizer'])

            for state in optimiser.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

        banner = "================================================================================"
        title = " Multi Task Training "

        print(banner)
        print(title)
        print(banner)