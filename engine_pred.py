import torch
import util.utils as utils
import numpy as np
from torchvision.utils import save_image
import os
import random

def other_maps(dataset):
    """Sample 10 reference fix maps from dataset for s-AUC"""
    while True:
        # this_map = np.zeros((320,640))
        for i in range(10):
            idx = random.randint(0,len(dataset)-1)
            frame_nr = random.randint(
                0, 15)
            _, _, fix_maps, _, _ = dataset[idx]
            this_this_map = fix_maps[frame_nr].numpy()
            if i==0:
                this_map = this_this_map 
            else:
                this_map = this_map + this_this_map

        this_map = np.clip(this_map, 0, 1)
        yield this_map

def test(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    dataset,
    device: torch.device,
    loss_metrics=('kld', 'nss', 'cc','sim','sAUC'),
    loss_weights=(1, -0.1, -0.1),
    frame_modulo=None,
    save_path=None,
):
    model.train(False)

    loss_epoch = []
    kld_epoch = []
    nss_epoch = []
    cc_epoch = []
    auc_epoch = []
    sim_epoch=[]

    maps=other_maps(dataset)
    for step, (frames, sals, fixs,video_num, start_idx) in enumerate(data_loader):
        frames = frames.to(device, non_blocking=True)
        sals = sals.to(device, non_blocking=True)
        fixs = fixs.to(device, non_blocking=True)

        # forward
        with torch.no_grad():
            output = model(frames)

        # save output as jpg
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                out = output[i,j,:,:].squeeze().squeeze()
                if not os.path.exists(save_path+str(video_num[i].item()).zfill(4)):
                    os.mkdir(save_path+str(video_num[i].item()).zfill(4))
                if video_num[i].item()>550:
                    #mirror out left-right
                    out=torch.flip(out,[1])
                save_image(out.exp()/(out.exp()).max(),save_path+str(video_num[i].item()).zfill(4)+"/"+str(start_idx[i].item()+j*frame_modulo).zfill(4)+".jpg")

        # loss
        losses=[]
        for i, metric in enumerate(loss_metrics):
            if metric == 'kld':
                loss_i = utils.kld_loss(output, sals)

            elif metric == 'nss':
                loss_i = utils.nss(output.exp(), fixs)
            elif metric == 'cc':
                loss_i = utils.corr_coeff(output.exp(), sals)
            elif metric == 'sim':
                sim = utils.similarity(output.exp(), sals)
                continue
            # elif metric == 'AUC_J':
            #     auc = []
            #     for i in range(output.shape[0]):
            #         for j in range(output.shape[1]):
            #             auc.append(utils.auc_judd((output.exp())[i,j,:,:].squeeze(), fixs[i,j,:,:].squeeze()))
            #     auc = np.mean(auc)
            #     continue
            elif metric == 'sAUC':
                auc = []
                map = next(maps)
                for i in range(output.shape[0]):
                    for j in range(output.shape[1]):
                        auc.append(utils.auc_shuff_acl((output.exp())[i,j,:,:].squeeze(), fixs[i,j,:,:].squeeze(),map))
                auc = np.mean(auc)
                continue
            else:
                raise NotImplementedError
            losses.append(loss_i.mean(1).mean(0))
        
        loss = sum([loss_i * weight for loss_i, weight in zip(losses, loss_weights)])
        loss_epoch.append(loss.item())

        kld_epoch.append(losses[0].item())
        nss_epoch.append(losses[1].item())
        cc_epoch.append(losses[2].item())
        # auc_epoch.append(auc)
        sim_epoch.append(sim)
        auc_epoch.append(auc)


    # print
    loss=np.mean(loss_epoch)
    kld=np.mean(kld_epoch)
    nss=np.mean(nss_epoch)
    cc=np.mean(cc_epoch)
    sim=np.mean(sim_epoch)
    auc=np.mean(auc_epoch)

    print(
        f"test\t"
        f"Loss: {loss:.4f}\t"
        f"kld: {kld:.4f}\t"
        f"nss: {nss:.4f}\t"
        f"cc: {cc:.4f}\t"
        f"sim: {sim:.4f}\t"
        f"AUC_J: {auc:.4f}\t"
    )

    return loss