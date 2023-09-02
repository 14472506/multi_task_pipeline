"""
Detials
"""
# imports
import torch
from torch.cuda.amp import autocast
import gc

# class
class Step():
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

    def _classifier_action(self, model, train_loader, val_loader, loss, optimiser, device, grad_acc, epoch, log, iter_count, logger):
        """ Detials """
        def train(model, loader, loss_fun, optimiser, device, scaler, grad_acc, epoch, log, iter_count, logger):
            """ Detials """
            # loop execution setup
            model.train()
            pf_loss = 0
            loss_acc = 0

            # loop
            for i, data in enumerate(loader):
                # get batch
                input, target = data
                input, target = input.to(device), target.to(device)

                with autocast():
                    output = model.forward(input)
                    loss = loss_fun(output, target)
            
                scaler.scale(loss).backward()
                if grad_acc:
                    if (i+1) % grad_acc == 0:
                        scaler.step(optimiser)
                        scaler.update()
                        optimiser.zero_grad()
                else:
                    scaler.step(optimiser)
                    scaler.update()
                    optimiser.zero_grad()

                # recording
                loss_val = loss.item()
                pf_loss += loss_val
                loss_acc += loss_val

                # reporting
                pf_loss = logger.train_loop_reporter(epoch, iter_count, device, pf_loss)
                iter_count += 1

            # logging goes here
            log["epochs"].append(epoch)
            log["train_loss"].append(loss_acc/len(loader))

            # returns

        def validate(model, loader, loss_fun, device , epoch, log, logger):
            """ Detials """
            # loop execution setup
            model.eval()
            loss_acc = 0

            for i, data in enumerate(loader):
                # get batch
                input, target = data
                input, target = input.to(device), target.to(device)

                with torch.no_grad():
                    output = model(input)
                    loss = loss_fun(output, target)

                loss_acc += loss.item()

            loss = loss_acc/len(loader)
            logger.val_loop_reporter(epoch, device, loss)
            log["val_loss"].append(loss)
                
        # initial params
        banner = "--------------------------------------------------------------------------------"
        train_title = "Training"
        val_title = "Validating"
        scaler = torch.cuda.amp.GradScaler(enabled=True)

        print(banner)
        print(train_title)
        print(banner)

        train(model, train_loader, loss, optimiser, device, scaler, grad_acc, epoch, log, iter_count, logger)

        print(banner)
        print(val_title)
        print(banner)

        validate(model, val_loader, loss, device, epoch, log, logger)
    


    
