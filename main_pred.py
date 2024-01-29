from model_uen import UEN
import torch
import argparse
from util.thue_dataset import THUEDataset
from engine_pred import test

def get_args_parser():
    parser = argparse.ArgumentParser(
        description="train uEn", add_help=False
    )

    parser.add_argument(
        "--model_path",
        default="./runs/uen_120/2024-01-12-01-37/epoch_4.pth",
        help="best model"
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

if __name__ == "__main__":
    args=args=get_args_parser().parse_args()
    net=UEN()
    net=net.to(args.device)

    # load model
    checkpoint=torch.load(args.model_path)
    net.load_state_dict(checkpoint)

    # load dataset
    dataset = THUEDataset(
        mode="test",
        num_frames=16,
        frame_modulo=5,
        videos_path="D:/00Dataset/THUE0001/annotation/",
        mean=(0.45, 0.45, 0.45),
        std=(0.225, 0.225, 0.225),
    )

    # load dataloader
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=10,
        pin_memory=True,
    )

    # test
    test(
        net,
        data_loader,
        dataset,
        args.device,
        loss_metrics=('kld', 'nss', 'cc','sim','sAUC'),
        loss_weights=(1, -0.1, -0.1),
        frame_modulo=5,
        save_path="D:/00Dataset/THUE0001/prediction/",
    )

    