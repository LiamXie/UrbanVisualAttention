from model_uen import UEN
import torch
import argparse
from util import utils
from util.thue_dataset import THUEDataset
import random
import numpy as np

from torchvision.utils import save_image
import os
from PIL import Image

def get_args_parser():
    parser = argparse.ArgumentParser(
        description="train uEn", add_help=False
    )

    parser.add_argument(
        "--model_path",
        default="./runs/uen_120/2024-01-12-01-37/epoch_4.pth",
        help="best model"
    )

    #save path
    parser.add_argument(
        "--save_path",
        default="E:/00论文/01Data/dataProc/00_PaperResult/human_annotation/model",
        help="save path"
    )

    # device
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )

    # Training parameters
    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument(
        "--lr",
        default=5e-3,
        type=float,
        help="learning rate",
    )
    parser.add_argument(
        "--layerwise_lr_decay",
        default=True,
        type=bool,
        help="layerwise_lr_decay",
    )
    parser.add_argument(
        "--layerwise_lr_decay_rate",
        default=0.01,
        type=float,
        help="layerwise_lr_decay_rate",
    )
    parser.add_argument(
        "--train_cnn_after",
        default=0,
        type=int,
        help="train cnn after # epoch",
    )
    parser.add_argument(
        "--grad_clip",
        default=2,
        type=float,
        help="gradient clip",
    )
    parser.add_argument(
        "--num_epochs",
        default=7,
        type=int,
        help="number of total epochs to run",
    )

    # log
    parser.add_argument(
        "--print_freq",
        default=50,
        type=int,
        help="print frequency",
    )
    return parser


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

if __name__ == "__main__":
    args=args=get_args_parser().parse_args()
    net=UEN()
    net=net.to(args.device)

    # load model
    checkpoint=torch.load(args.model_path)
    net.load_state_dict(checkpoint)

    net.train(False)

    # load dataset
    dataset = THUEDataset(
        mode="test",
        num_frames=16,
        frame_modulo=5,
        videos_path="D:/00Dataset/THUE0001/annotation/",
        mean=(0.45, 0.45, 0.45),
        std=(0.225, 0.225, 0.225),
    )

    # load specific sequence
    video_nums = [544,433,422,398,511,517,453,421]
    target_frames = [135,220,34,63,39,249,125,85]
    frame_starts=[]
    frame_ids=[]
    for frame in target_frames:
        for i in range(1,16):
            if i==15:
                frame_starts.append(frame-5*15)
                frame_ids.append(15)
                break
            if frame-5*i>0:
                continue
            else:
                frame_starts.append(frame-5*(i-1))
                frame_ids.append(i-1)
                break
        
    print(frame_starts, frame_ids)

    NSS=[]
    CC=[]
    SIM=[]
    sAUC=[]
    maps = other_maps(dataset)

    NSS_human=[[] for _ in range(5)]
    CC_human=[[] for _ in range(5)]
    SIM_human=[[] for _ in range(5)]
    sAUC_human=[[] for _ in range(5)]

    NSS_all=[]
    CC_all=[]
    SIM_all=[]
    sAUC_all=[]


    for idx,(video_num, frame_start, frame_id) in enumerate(zip(video_nums, frame_starts, frame_ids)):
        frames, sals, fixs = dataset.load_specific_sq(video_num, frame_start)
        frames = frames.unsqueeze(0)
        sals = sals.unsqueeze(0)
        fixs = fixs.unsqueeze(0)
        frames = frames.to(args.device)
        sals = sals.to(args.device)
        fixs = fixs.to(args.device)

        sal = sals[:,frame_id,:,:]
        sal = sal.unsqueeze(1)
        fix = fixs[:,frame_id,:,:]
        fix = fix.unsqueeze(1)

        with torch.no_grad():
            preds = net(frames)
        pred = preds[:,frame_id,:,:]
        # save pred
        save_image(pred.exp()/(pred.exp()).max(), os.path.join(args.save_path, str(idx).zfill(4)+".jpg"))
        pred = pred.unsqueeze(1)
        NSS_=utils.nss(pred.exp(), fix)
        NSS.append(NSS_.item())
        CC_=utils.corr_coeff(pred.exp(), sal)
        CC.append(CC_.item())
        SIM_=utils.similarity(pred.exp(), sal)
        SIM.append(SIM_.item())
        sAUC.append(utils.auc_shuff_acl(pred.exp().squeeze(), fix.squeeze(),next(maps)))

        for id_huam in range(1,6):
            pred = Image.open("E:/00论文/01Data/dataProc/00_PaperResult/human_annotation/arthitects/"+str(id_huam).zfill(2)+"/sal/0001/"+str(idx+1).zfill(3)+".png").convert('L')
            pred = pred.resize((480,270))
            pred = np.array(pred)
            pred = pred/255
            pred = torch.from_numpy(pred)
            pred = utils.prob_tensor(pred)
            pred = pred.to(args.device)
            pred = pred.unsqueeze(0)
            pred = pred.unsqueeze(0)
            NSS_=utils.nss(pred, fix)
            NSS_human[id_huam-1].append(NSS_.item())
            CC_=utils.corr_coeff(pred, sal)
            CC_human[id_huam-1].append(CC_.item())
            SIM_=utils.similarity(pred, sal)
            SIM_human[id_huam-1].append(SIM_.item())
            sAUC_human[id_huam-1].append(utils.auc_shuff_acl(pred.squeeze(), fix.squeeze(),next(maps)))
        
        pred = Image.open("E:/00论文/01Data/dataProc/00_PaperResult/human_annotation/arthitects/all/sal/"+str(idx+1).zfill(3)+".png").convert('L')
        pred = pred.resize((480,270))
        pred = np.array(pred)
        pred = pred/255
        pred = torch.from_numpy(pred)
        pred = utils.prob_tensor(pred)
        pred = pred.to(args.device)
        pred = pred.unsqueeze(0)
        pred = pred.unsqueeze(0)
        NSS_=utils.nss(pred, fix)
        NSS_all.append(NSS_.item())
        CC_=utils.corr_coeff(pred, sal)
        CC_all.append(CC_.item())
        SIM_=utils.similarity(pred, sal)
        SIM_all.append(SIM_.item())
        sAUC_all.append(utils.auc_shuff_acl(pred.squeeze(), fix.squeeze(),next(maps)))
    



    nss=np.mean(NSS)
    cc=np.mean(CC)
    sim=np.mean(SIM)
    sauc=np.mean(sAUC)
    print(
        f"NSS: {nss:.4f}\t"
        f"CC: {cc:.4f}\t"
        f"SIM: {sim:.4f}\t"
        f"sAUC: {sauc:.4f}\t"
    )

    for id_huam in range(1,6):
        nss=np.mean(NSS_human[id_huam-1])
        cc=np.mean(CC_human[id_huam-1])
        sim=np.mean(SIM_human[id_huam-1])
        sauc=np.mean(sAUC_human[id_huam-1])
        print(
            f"NSS_human_{id_huam}: {nss:.4f}\t"
            f"CC_human_{id_huam}: {cc:.4f}\t"
            f"SIM_human_{id_huam}: {sim:.4f}\t"
            f"sAUC_human_{id_huam}: {sauc:.4f}\t"
        )

    nss=np.mean(NSS_all)
    cc=np.mean(CC_all)
    sim=np.mean(SIM_all)
    sauc=np.mean(sAUC_all)
    print(
        f"NSS_all: {nss:.4f}\t"
        f"CC_all: {cc:.4f}\t"
        f"SIM_all: {sim:.4f}\t"
        f"sAUC_all: {sauc:.4f}\t"
    )