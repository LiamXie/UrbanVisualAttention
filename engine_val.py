import torch
import util.utils as utils
import numpy as np

def val_one_epoch(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    loss_metrics=('kld', 'nss', 'cc'),
    loss_weights=(1, -0.1, -0.1),
):
    model.train(False)

    loss_epoch = []
    kld_epoch = []
    nss_epoch = []
    cc_epoch = []
    auc_epoch = []
    for step, (frames, sals, fixs) in enumerate(data_loader):
        frames = frames.to(device, non_blocking=True)
        sals = sals.to(device, non_blocking=True)
        fixs = fixs.to(device, non_blocking=True)

        # forward
        with torch.no_grad():
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
            # elif metric == 'AUC_J':
            #     auc = utils.auc_judd(output.exp(), fixs)
            else:
                raise NotImplementedError
            losses.append(loss_i.mean(1).mean(0))
        
        loss = sum([loss_i * weight for loss_i, weight in zip(losses, loss_weights)])
        loss_epoch.append(loss.item())

        kld_epoch.append(losses[0].item())
        nss_epoch.append(losses[1].item())
        cc_epoch.append(losses[2].item())
        # auc_epoch.append(auc)


    # print
    loss=np.mean(loss_epoch)
    kld=np.mean(kld_epoch)
    nss=np.mean(nss_epoch)
    cc=np.mean(cc_epoch)
    print(
        f"Epoch: [{epoch+1}]val\t"
        f"Loss: {loss:.4f}\t"
        f"kld: {kld:.4f}\t"
        f"nss: {nss:.4f}\t"
        f"cc: {cc:.4f}\t"
        # f"AUC_J: {auc:.4f}\t"
    )

    return loss