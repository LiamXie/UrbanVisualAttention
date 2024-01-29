import torch
import util.utils as utils
import matplotlib.pyplot as plt
import numpy as np

def train_one_epoch(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    train_cnn_after: int,
    loss_metrics=('kld', 'nss', 'cc'),
    loss_weights=(1, -0.1, -0.1),
    args=None,
    fp32=False,
):
    model.train(True)

    cnn_grad=epoch>=train_cnn_after
    for param in model.cnn.parameters():
        param.requires_grad=cnn_grad

    loss_epoch = []
    for step, (frames, sals, fixs) in enumerate(data_loader):
        frames = frames.to(device, non_blocking=True)
        sals = sals.to(device, non_blocking=True)
        fixs = fixs.to(device, non_blocking=True)

        # forward
        output = model(frames)

        # loss
        losses=[]
        for i, metric in enumerate(loss_metrics):
            if metric == 'kld':
                loss_i = utils.kld_loss(output, sals)

            elif metric == 'nss':
                loss_i = utils.nss(output.exp(), fixs)
            elif metric == 'cc':
                loss_i = utils.corr_coeff(output.exp(), sals)
            else:
                raise NotImplementedError
            losses.append(loss_i.mean(1).mean(0))
        
        loss = sum([loss_i * weight for loss_i, weight in zip(losses, loss_weights)])
        # backward
        optimizer.zero_grad()
        loss.backward()
        # 实施梯度裁剪
        if args.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip)
        optimizer.step()
        loss_epoch.append(loss.item())


    # print
    loss=np.mean(loss_epoch)
    print(
        f"Epoch: [{epoch+1}][{step+1}/{len(data_loader)}]train\t"
        f"Loss: {loss:.4f}\t"
        # f"kld: {losses[0].item():.4f}\t"
        # f"nss: {losses[1].item():.4f}\t"
        # f"cc: {losses[2].item():.4f}\t"
    )

    return loss    