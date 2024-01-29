import torch
import util.utils as utils
import numpy as np
import os
from PIL import Image
import random
import cv2

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

def baseline_test(
    data_loader: torch.utils.data.DataLoader,
    data_set,
    device: torch.device,
    loss_metrics=('kld', 'nss', 'cc','sim','sAUC'),
    loss_weights=(1, -0.1, -0.1),
    bias: torch.Tensor=None,
):
    maps=other_maps(data_set)
    loss_epoch = []
    kld_epoch = []
    nss_epoch = []
    cc_epoch = []
    auc_epoch = []
    sim_epoch=[]
    for step, (_, sals, fixs,_,_) in enumerate(data_loader):
        print(step)
        sals = sals.to(device, non_blocking=True)
        fixs = fixs.to(device, non_blocking=True)

        # forward

        # loss
        losses=[]
        for i, metric in enumerate(loss_metrics):
            if metric == 'kld':
                loss_i = utils.kld_loss(bias.log(), sals)

            elif metric == 'nss':
                loss_i = utils.nss(bias, fixs)
            elif metric == 'cc':
                loss_i = utils.corr_coeff(bias, sals)
            elif metric == 'sim':
                sim = utils.similarity(bias, sals)
                continue
            # elif metric == 'AUC_J':
            #     auc = []
            #     for i in range(bias.shape[0]):
            #         for j in range(bias.shape[1]):
            #             auc.append(utils.auc_judd((bias.exp())[i,j,:,:].squeeze(), fixs[i,j,:,:].squeeze()))
            #     auc = np.mean(auc)
            #     continue
            elif metric == 'sAUC':
                auc = []
                map=next(maps)
                for i in range(bias.shape[0]):
                    for j in range(bias.shape[1]):
                        auc.append(utils.auc_shuff_acl(bias[i,j,:,:].squeeze(), fixs[i,j,:,:].squeeze(),map))
                auc = np.mean(auc)
                continue
            losses.append(loss_i.mean(1).mean(0))

        
        loss = sum([loss_i * weight for loss_i, weight in zip(losses[0:3], loss_weights[0:3])])
        loss_epoch.append(loss.item())

        kld_epoch.append(losses[0].item())
        nss_epoch.append(losses[1].item())
        cc_epoch.append(losses[2].item())
        auc_epoch.append(auc)
        sim_epoch.append(sim)


    # print
    loss=np.mean(loss_epoch)
    kld=np.mean(kld_epoch)
    nss=np.mean(nss_epoch)
    cc=np.mean(cc_epoch)
    sim=np.mean(sim_epoch)
    auc=np.mean(auc_epoch)
    print(
        f"Baseline\t"
        f"Loss: {loss:.4f}\t"
        f"kld: {kld:.4f}\t"
        f"nss: {nss:.4f}\t"
        f"cc: {cc:.4f}\t"
        f"sAUC: {auc:.4f}\t"
        f"sim: {sim:.4f}\t"
    )

    return loss

def pred_360_test(
    data_loader: torch.utils.data.DataLoader,
    data_set,
    device: torch.device,
    loss_metrics=('kld', 'nss', 'cc','sim','sAUC'),
    loss_weights=(1, -0.1, -0.1),
    pred_path="",
):
    maps=other_maps(data_set)
    num_frames=16
    modulo=5
    loss_epoch = []
    kld_epoch = []
    nss_epoch = []
    cc_epoch = []
    auc_epoch = []
    sim_epoch=[]
    for _, (_, sals, fixs,video_nums, start_idxs) in enumerate(data_loader):
        # print(step)
        sals = sals.to(device, non_blocking=True)
        fixs = fixs.to(device, non_blocking=True)

        preds=[]
        for video_num, start_idx in zip(video_nums, start_idxs):
            seq=[]
            for i in range(start_idx,start_idx+num_frames*modulo,modulo):
                # load jpg with gray scale
                pred=np.array(Image.open(os.path.join(pred_path,str(video_num.item()).zfill(4),str(i).zfill(4)+".jpg")).convert('L'))
                pred=pred/255
                pred=torch.from_numpy(pred)
                pred=pred.to(device, non_blocking=True)
                pred=utils.prob_tensor(pred)
                seq.append(pred)
            seq=torch.stack(seq)
            preds.append(seq)
        
        preds=torch.stack(preds)        
                
        # loss
        losses=[]
        for i, metric in enumerate(loss_metrics):
            if metric == 'kld':
                loss_i = utils.kld_loss(preds.log(), sals)

            elif metric == 'nss':
                loss_i = utils.nss(preds, fixs)
            elif metric == 'cc':
                loss_i = utils.corr_coeff(preds, sals)
            elif metric == 'sim':
                sim = utils.similarity(preds, sals)
                continue
            # elif metric == 'AUC_J':
            #     auc = []
            #     for i in range(preds.shape[0]):
            #         for j in range(preds.shape[1]):
            #             auc.append(utils.auc_judd(preds[i,j,:,:].squeeze(), fixs[i,j,:,:].squeeze()))
            #     auc = np.mean(auc)
            #     continue
            elif metric == 'sAUC':
                auc = []
                map=next(maps)
                for i in range(preds.shape[0]):
                    for j in range(preds.shape[1]):
                        auc.append(utils.auc_shuff_acl(preds[i,j,:,:].squeeze(), fixs[i,j,:,:].squeeze(),map))
                auc = np.mean(auc)
                continue
            losses.append(loss_i.mean(1).mean(0))

        
        loss = sum([loss_i * weight for loss_i, weight in zip(losses[0:3], loss_weights[0:3])])
        loss_epoch.append(loss.item())

        kld_epoch.append(losses[0].item())
        nss_epoch.append(losses[1].item())
        cc_epoch.append(losses[2].item())
        auc_epoch.append(auc)
        sim_epoch.append(sim)


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
        f"sAUC: {auc:.4f}\t"
        f"sim: {sim:.4f}\t"
    )

    return loss