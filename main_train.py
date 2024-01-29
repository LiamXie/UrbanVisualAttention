import argparse
import torch
from util.thue_dataset import THUEDataset
from model_uen import UEN

from engine_train import train_one_epoch    
from engine_val import val_one_epoch

import datetime

import os

def get_args_parser():
    parser = argparse.ArgumentParser(
        description="train uEn", add_help=False
    )

    # dataset folder
    parser.add_argument(
        "--dataset_folder", default="D:/00Dataset/THUE0001/", help="dataset folder"
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

def train():
    args=get_args_parser().parse_args()

    # load dataset
    dataset = THUEDataset(
        videos_path=args.dataset_folder+"annotation/",
        mode="train",
        num_frames=16,
        frame_modulo=5, # select frames every num_steps frames
        # norm setting
        mean=(0.45, 0.45, 0.45),
        std=(0.225, 0.225, 0.225),
        target_size=None,
    )

    dataset_val = THUEDataset(
        videos_path=args.dataset_folder+"annotation/",
        mode="val",
        num_frames=16,
        frame_modulo=5, # select frames every num_steps frames
        # norm setting
        mean=(0.45, 0.45, 0.45),
        std=(0.225, 0.225, 0.225),
        target_size=None,
    )

    data_loader_train=torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=10,
        pin_memory=True,
        drop_last=True,
        shuffle=True
    )

    data_loader_val=torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=10,
        pin_memory=True,
        drop_last=True,
        shuffle=True
    )
    
    # load model
    net=UEN()
    net=net.to(args.device)

    # optimizer
    # layer-wise learning rate and weight decay
    if args.layerwise_lr_decay:
        params = []
        for key, value in dict(net.named_parameters()).items():
            if value.requires_grad:
                if 'cnn' in key:
                    params += [{'params': [value], 'lr': args.lr*args.layerwise_lr_decay_rate}]
                else:
                    params += [{'params': [value], 'lr': args.lr}]
        optimizer = torch.optim.AdamW(params)#, lr=args.lr)
        print("layer-wise learning rate and weight decay")
    else:
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr)
        print("no layer-wise learning rate and weight decay")


    now = datetime.datetime.now()
    folder_name = now.strftime("%Y-%m-%d-%H-%M")
    os.mkdir("./runs/"+folder_name)

    # create csv file
    with open("./runs/"+folder_name+"/log.csv","w") as f:
        f.write("epoch,train_loss,val_loss\n")

    # save args
    with open("./runs/"+folder_name+"/args.txt","w") as f:
        f.write(str(args))

    # collect initial loss
    loss_train=val_one_epoch(
        model=net,
        data_loader=data_loader_train,\
        device=args.device,
        epoch=-1,
    )

    loss_val=val_one_epoch(
        model=net,
        data_loader=data_loader_val,
        device=args.device,
        epoch=-1,
    )

    # write log
    with open("./runs/"+folder_name+"/log.csv","a") as f:
        f.write("{},{},{}\n".format(0,loss_train,loss_val))
    print("initial loss: train_loss:{}, val_loss:{}".format(loss_train,loss_val))

    for epoch in range(args.num_epochs):
        loss_train=train_one_epoch(
            model=net,
            data_loader=data_loader_train,
            optimizer=optimizer,
            device=args.device,
            epoch=epoch,
            train_cnn_after=args.train_cnn_after,
            args=args,
            fp32=False,)
        loss_val=val_one_epoch(
            model=net,
            data_loader=data_loader_val,
            device=args.device,
            epoch=epoch,
        )
        # save model
        torch.save(net.state_dict(), "./runs/"+folder_name+"/epoch_{}.pth".format(epoch+1))

        # write log
        with open("./runs/"+folder_name+"/log.csv","a") as f:
            f.write("{},{},{}\n".format(epoch+1,loss_train,loss_val))

if __name__ =="__main__":
    for i in range(20):
        train()