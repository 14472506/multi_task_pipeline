def train(model, loader, loss_fun, optimiser, device, scaler, grad_acc, epoch, log, iter_count, logger):
    model.train()
    sup_iter = iter(loader[0])

    awl = loss_fun[0]
    loss = loss_fun[1]

    primary_grad = grad_acc[0] if grad_acc else 1
    secondar_grad = grad_acc[1] if grad_acc else 1

    accumulated_loss = 0

    for i in range(len(loader[0])):
        sup_im, sup_target, sup_ssl_target = next(sup_iter)
        if self.model_name == "jigmask_multi_task":
            sup_im = sup_im[0].to(device)
        else:
            sup_im = [image.to(device) for image in sup_im]
        sup_target = [{k: v.to(device) for k, v in t.items()} for t in sup_target]
        sup_ssl_target = sup_ssl_target[0].to(device)

        with autocast():
            sup_output, sup_ssl_pred = model.forward(sup_im, sup_target)
            sup_loss = sum(loss for loss in sup_output.values())
            ssl_loss = loss(sup_ssl_pred, sup_ssl_target.unsqueeze(0))

            ssl_loss_accumulated = 0
            for _ in range(secondar_grad):
                try:
                    ssl_im, ssl_target = next(self.train_ssl_iter)
                except StopIteration:
                    self.train_ssl_iter = iter(loader[1])
                    ssl_im, ssl_target = next(self.train_ssl_iter)
                ssl_im = ssl_im.to(device)
                ssl_target = ssl_target.to(device)

                ssl_output = model.forward(ssl_im)
                ssl_loss_accumulated += loss(ssl_output, ssl_target)

            ssl_loss_total = (ssl_loss + ssl_loss_accumulated / secondar_grad) / primary_grad
            accumulated_loss += ssl_loss_total.item()

            weighted_losses = awl(sup_output["loss_classifier"], sup_output["loss_box_reg"], sup_output["loss_mask"], sup_output["loss_objectness"], sup_output["loss_rpn_box_reg"], ssl_loss_total)
            scaler.scale(weighted_losses).backward()

        if (i + 1) % primary_grad == 0:
            scaler.step(optimiser)
            scaler.update()
            optimiser.zero_grad()

        if i % 10 == 0:  # Clear GPU memory less frequently
            torch.cuda.empty_cache()

        iter_count += 1

    logger.update_loss_stats(accumulated_loss / len(loader[0]), epoch)