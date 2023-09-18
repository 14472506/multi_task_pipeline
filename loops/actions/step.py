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
            "rotnet_resnet_50": self._classifier_action,
            "rotmask_multi_task": self._multitask_action
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
    
    def _multitask_action(self, model, train_loader, val_loader, loss, optimiser, device, grad_acc, epoch, log, iter_count, logger):
        """ Detials """
        def train(model, loader, loss_fun, optimiser, device, scaler, grad_acc, epoch, log, iter_count, logger):
            """ Detials """
            # loop execution setup
            model.train()

            # supervised and self supervised loader extraction
            sup_iter = iter(loader[0])
            ssl_iter = iter(loader[1])

            # losses
            awl = loss_fun[0]
            loss = loss_fun[1]

            primary_grad = grad_acc[0] if grad_acc else 1
            secondar_grad = grad_acc[1] if grad_acc else 1

            sup_loss_acc, sup_ssl_loss_acc, ssl_loss_acc, weighted_losses_acc, pf_loss = 0, 0, 0, 0, 0

            # ssl step adjust goes here
            ssl_adjust = log["iter_accume"] % len(loader[1])
            if ssl_adjust:
                print("Adjusting by %s steps" %(ssl_adjust))
                for i in range(ssl_adjust):
                    _, _ = next(ssl_iter)
                
            for i in range(len(loader[0])):
                sup_im, sup_target, sup_ssl_im, sup_ssl_target = next(sup_iter)
                sup_im = list(image.to(device) for image in sup_im)
                sup_target = [{k: v.to(device) for k, v in t.items()} for t in sup_target]
                sup_ssl_im = sup_ssl_im[0].to(device)
                sup_ssl_target = sup_ssl_target[0].to(device)

                with autocast():
                    # forward pass
                    sup_output = model.forward(sup_im, sup_target, action="supervised")
                    sup_loss = sum(loss for loss in sup_output.values())
                    sup_ssl_output = model.forward(sup_ssl_im, action="self_supervised")
                    sup_ssl_loss = loss(sup_ssl_output[0], sup_ssl_target)

                    #sup_loss /= primary_grad
                    #sup_ssl_loss /= primary_grad

                    sup_loss_acc += sup_loss.item()
                    sup_ssl_loss_acc += sup_ssl_loss.item()
                    
                    for i in range(0, secondar_grad):
                        try:
                            ssl_im, ssl_target = next(ssl_iter) 
                        except StopIteration:
                            print("resetting iter")
                            ssl_iter = iter(loader[1])
                            ssl_im, ssl_target = next(ssl_iter)
                        ssl_im = ssl_im.to(device)
                        ssl_target = ssl_target.to(device)

                        # forward pass
                        ssl_output = model.forward(ssl_im, action="self_supervised")
                        ssl_loss = loss(ssl_output, ssl_target)

                        ssl_loss /= secondar_grad

                    ssl_loss_acc += ssl_loss.item()
                    weighted_losses = awl(sup_loss, sup_ssl_loss, ssl_loss)
                    weighted_losses_acc += weighted_losses.item()
                    pf_loss += weighted_losses.item()
                
                scaler.scale(weighted_losses).backward()
                if (i+1) % primary_grad == 0:
                    # optimiser step
                    scaler.step(optimiser)
                    scaler.update()
                    optimiser.zero_grad()
                                    
                # reporting
                pf_loss = logger.train_loop_reporter(epoch, iter_count, device, pf_loss)
                iter_count += 1

            # accumulating iter count for ssl iter adjust
            log["iter_accume"] += iter_count*secondar_grad

            # logging
            log["epochs"].append(epoch)
            log["train_loss"].append(weighted_losses_acc/len(loader[0]))
            log["train_sup_loss"].append(sup_loss_acc/len(loader[0]))
            log["train_sup_ssl_loss"].append(sup_ssl_loss_acc/len(loader[0]))
            log["train_ssl_loss"].append(ssl_loss_acc/len(loader[0]))

        def validate(model, loader, loss_fun, device , epoch, log, logger):
            """ Detials """
            # loop execution setup
            model.train()
            sup_loss_acc = 0
            sup_ssl_loss_acc = 0
            ssl_loss_acc = 0
            weighted_losses_acc = 0

            # supervised and self supervised loader extraction
            sup_iter = iter(loader[0])
            ssl_iter = iter(loader[1])

            # losses
            awl = loss_fun[0]
            loss = loss_fun[1]

            # ssl step adjust goes here
            ssl_adjust = log["val_it_accume"] % len(loader[1])
            if ssl_adjust:
                print("Adjusting by %s steps" %(ssl_adjust))
                for i in range(ssl_adjust):
                    _, _ = next(ssl_iter)

            # ssl step adjust goes here
            for i in range(len(loader[0])):
                sup_im, sup_target, sup_ssl_im, sup_ssl_target = next(sup_iter)
                sup_im = list(image.to(device) for image in sup_im)
                sup_target = [{k: v.to(device) for k, v in t.items()} for t in sup_target]
                sup_ssl_im = sup_ssl_im[0].to(device)
                sup_ssl_target = sup_ssl_target[0].to(device)

                try:
                    ssl_im, ssl_target = next(ssl_iter) 
                except StopIteration:
                    print("resetting iter")
                    ssl_iter = iter(loader[1])
                    ssl_im, ssl_target = next(ssl_iter)  
                ssl_im = ssl_im.to(device)
                ssl_target = ssl_target.to(device)

                with torch.no_grad():
                    with autocast():
                        # forward pass
                        sup_output = model.forward(sup_im, sup_target, action="supervised")
                        sup_loss = sum(loss for loss in sup_output.values())
                        sup_ssl_output = model.forward(sup_ssl_im, action="self_supervised")
                        sup_ssl_loss = loss(sup_ssl_output[0], sup_ssl_target)
                        ssl_output = model.forward(ssl_im, action="self_supervised")
                        ssl_loss = loss(ssl_output, ssl_target)
                
                        weighted_losses = awl(sup_loss, sup_ssl_loss, ssl_loss)

                # collecting losses
                sup_loss_acc += sup_loss.item()
                sup_ssl_loss_acc += sup_ssl_loss.item()
                ssl_loss_acc += ssl_loss.item()
                weighted_losses_acc += weighted_losses.item()
            
            # adjusting val iter accumulation for ssl step adjust
            log["val_it_accume"] += len(loader[0])

            # logging
            log["val_loss"].append(weighted_losses_acc/len(loader[0]))
            log["val_sup_loss"].append(sup_loss_acc/len(loader[0]))
            log["val_sup_ssl_loss"].append(sup_ssl_loss_acc/len(loader[0]))
            log["val_ssl_loss"].append(ssl_loss_acc/len(loader[0]))
            
            logger.val_loop_reporter(epoch, device, log["val_sup_loss"][-1])

        # initial params
        banner = "--------------------------------------------------------------------------------"
        train_title = "Training"
        val_title = "Validating"

        loss[0].to(device)
        scaler = torch.cuda.amp.GradScaler(enabled=True)

        print(banner)
        print(train_title)
        print(banner)

        train(model, train_loader, loss, optimiser, device, scaler, grad_acc, epoch, log, iter_count, logger)

        print(banner)
        print(val_title)
        print(banner)

        validate(model, val_loader, loss, device, epoch, log, logger)