import torch
from torch.utils.data import  DataLoader
from util.thue_dataset import THUEDataset
import os
import numpy as np
from PIL import Image
from util import utils
from engine_test import baseline_test

def load_bias(bias_path, batch_size, num_frames,target_size):
    img = Image.open(bias_path).convert("L")
    img = img.resize(target_size)
    img = np.array(img)
    img = torch.from_numpy(img)
    # img = img.float()
    img = img/255
    img = utils.prob_tensor(img)
    seq=[img for _ in range(num_frames)]
    seq=torch.stack(seq, dim=0)
    seq = [seq for _ in range(batch_size)]
    seq = torch.stack(seq, dim=0)
    return seq


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

    # print dataset len
    print("dataset len: ", len(data_loader_test))

    bias = load_bias("D:/00Dataset/THUE0001_360/bias.png", batch_size, num_frames,target_size=(640,320))

    bias = bias.to(torch.device("cuda"))

    baseline_test(data_loader_test, dataset,torch.device("cuda"), bias=bias)