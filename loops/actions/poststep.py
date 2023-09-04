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
        self.stepped = False
    
    def action(self):
        self.action_map = {
            "rotnet_resnet_50": self._classifier_action,
            "rotmask_multi_task": self._multitask_action
        }
        return self.action_map[self.model_name]

    def _classifier_action(self, epoch, model, optimiser, scheduler, logs, logger):
        """ Detials """
        logger.save_model(epoch, model, optimiser, "last")

        if logs["val_loss"][-1] <= logger.best:
            if self.stepped:
                logs["post_best_val"].append(logs["val_loss"][-1])
                logs["post_best_epoch"].append(epoch)
                logger.save_model(epoch, model, optimiser, "post")
                logger.best = logs["val_loss"][-1]
            else:
                logs["pre_best_val"].append(logs["val_loss"][-1])
                logs["pre_best_epoch"].append(epoch)
                logger.save_model(epoch, model, optimiser, "pre")
                logger.best = logs["val_loss"][-1]

        logger.update_log_file(logs)

        if scheduler:
            self._handle_scheduler_step(scheduler, epoch, model, logger)

    def _multitask_action(self, epoch, model, optimiser, scheduler, logs, logger):
        """ Detials """
        logger.save_model(epoch, model, optimiser, "last")

        if logs["val_loss"][-1] <= logger.best:
            if self.stepped:
                logs["post_best_val"].append(logs["val_sup_loss"][-1])
                logs["post_best_epoch"].append(epoch)
                logger.save_model(epoch, model, optimiser, "post")
                logger.best = logs["val_sup_loss"][-1]
            else:
                logs["pre_best_val"].append(logs["val_sup_loss"][-1])
                logs["pre_best_epoch"].append(epoch)
                logger.save_model(epoch, model, optimiser, "pre")
                logger.best = logs["val_sup_loss"][-1]

        logger.update_log_file(logs)

        if scheduler:
            self._handle_scheduler_step(scheduler, epoch, model, logger)
        
    def _handle_scheduler_step(self, scheduler, epoch, model, logger):
        """ Detials """
        if epoch == logger.step-1:
            logger.load_model(model, "pre")
            logger.best = float('inf')
            self.stepped = True
        scheduler.step()