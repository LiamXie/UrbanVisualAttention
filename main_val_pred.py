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
        videos_path="D:/00Dataset/THUE0001/annotation/",
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

    pred_360_test(data_loader_test, dataset, torch.device("cuda"), pred_path="D:/00Dataset/THUE0001/prediction")