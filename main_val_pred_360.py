import torch
from torch.utils.data import  DataLoader
from util.thue_dataset import THUEDataset
import os
import numpy as np
from PIL import Image
from util import utils
from engine_test import pred_360_test



if __name__ == "__main__":
    batch_size = 1
    num_frames = 16
    dataset = THUEDataset(
        videos_path="D:/00Dataset/THUE0001_360/THUE0001_ERP/annotation/",
        mode="test",
        num_frames=num_frames,
        frame_modulo=5, # select frames every num_steps frames
        # norm setting
        mean=(0.45, 0.45, 0.45),
        std=(0.225, 0.225, 0.225),
        target_size=None,
    )
    data_loader_test=DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=10,
        pin_memory=True,
    )

    for alpha in np.arange(0.3,0.8,0.1):
        print(alpha)
        pred_360_test(data_loader_test, dataset, torch.device("cuda"), pred_path="D:/00Dataset/THUE0001_360/prediction/{:.1f}/".format(alpha))
    # # pred_360_test(data_loader_test, torch.device("cuda"), pred_path="D:/00Dataset/THUE0001_360/prediction/0.4/")
    print("dot")
    pred_360_test(data_loader_test, torch.device("cuda"), pred_path="D:/00Dataset/THUE0001_360/prediction/dot/")
    print("ERP")
    pred_360_test(data_loader_test, dataset,torch.device("cuda"), pred_path="D:/00Dataset/THUE0001_360/THUE0001_ERP/prediction")