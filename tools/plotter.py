"""
Module Detials:
This module uses the results from the training and test process to plot the results
of training for presentation
"""
# imports
# base packages

# third party packages

# local packages

# class
class Plotter():
    """ Detials """
    def __init__(self, cfg):
        self.cfg = cfg

        # extract config
        self._extract_config()

        # initialise something? what do i call this again?
        self._initialise_optimiser()
        self._initialise_logs

    def plotter(self):
        """ Detials """
        self.plotter_map = {
            "rotnet_resnet_50": self._classifier_plot,
            "jigsaw": self._classifier_plot,
            "mask_rcnn": self._instance_seg_plot,
            "rotmask_multi_task": self._multitask_plot
        }
        self.plotter_map[self.model_name]()

    def _extract_config(self):
        """ Details """
        self.model_name = self.cfg["model_name"]
        self.optimiser_cfg = self.cfg["optimiser"]
        self.logs_cfg = self.cfg["logs"]
    
    def _initialise_logs(self):
        """ Initialises the logger module for testing """
        self.logger = Logs(self.logs_cfg)

    def _initialise_optimiser(self):
        """
        Initialises step attribute for edentifying if a scheduler has been
        used in the training process. This is used for determining weather
        pre step of pre and post step models need to be evaluated.
        """
        self.step = True if self.optimiser_cfg["sched_name"] else False
    
    def _classifier_plot(self):
        """ Detials """
        print("implement classifier plotter")

    def _instance_seg_plot(self):
        """ Detials """
        print("implement instance seg plotter")

    def _multitask_plot(self):
        """ Details """
        print("implement multi task plotter")       
