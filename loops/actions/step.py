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

    def _classifier_action(self, model, train_loader, val_loader, loss, optimiser, device, grad_acc=None):
        """ Detials """
        def train(model, loader, loss_fun, optimiser, device, scaler, grad_acc=None):
            """ Detials """
            # loop execution setup
            model.train()

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
                        print(loss)
                else:
                    scaler.step(optimiser)
                    scaler.update()
                    optimiser.zero_grad()
                    print(loss)

                # recording

            # logging goes here

            # returns

        def validate(model, loader, loss_fun, device):
            """ Detials """
            # loop execution setup
            model.eval()

            for i, data in enumerate(loader):
                # get batch
                input, target = data
                input, target = input.to(device), target.to(device)

                with torch.no_grad():
                    output = model(input)
                    loss = loss_fun(output, target)

                print(loss)

        # initial params
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        train(model, train_loader, loss, optimiser, device, scaler, grad_acc)
        validate(model, val_loader, loss, device)
    


    
