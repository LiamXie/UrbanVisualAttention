import torch
import numpy as np
import os
from PIL import Image

from util import utils

# class of data loader, inherit from torch.utils.data.Dataset, reading video frame from folder
class THUEDataset(torch.utils.data.Dataset):
    def __init__(self, 
                videos_path,
                mode,
                num_frames=16,
                frame_modulo=5, # select frames every num_steps frames
                # norm setting
                mean=(0.45, 0.45, 0.45),
                std=(0.225, 0.225, 0.225),
                target_size=None,
                left_right=False
    ):
        self.videos_path = videos_path
        self.frame_modulo=frame_modulo
        self.mode = mode
        self.num_frames = num_frames
        self._mean = mean
        self._std = std
        self.target_size=target_size


        self.video_list_modified=[]

        self.left_right=left_right
        # separate the video list into train, val and test
        if mode == "train":
            self.video_list = [i for i in range(1, 351)] if not left_right else [i for i in range(1, 351)]+[j for j in range(1+550,351+550)]
            for video in self.video_list:
                self.video_list_modified=self.video_list_modified+\
                    ([video]*int(len(os.listdir(os.path.join(videos_path,str(video).zfill(4),"images")))/num_frames/self.frame_modulo+1))
            # shuffle the video list
            np.random.shuffle(self.video_list_modified)

        if mode == "val":
            self.video_list = [i for i in range(351, 386)]if not left_right else [i for i in range(351, 386)]+[j for j in range(351+550,386+550)]
            for video in self.video_list:
                self.video_list_modified=self.video_list_modified+\
                    ([video]*int(len(os.listdir(os.path.join(videos_path,str(video).zfill(4),"images")))/num_frames/self.frame_modulo+1))
            # shuffle the video list
            # np.random.shuffle(self.video_list_modified)

        elif mode == "test":
            self.video_list = [i for i in range(386, 551)] if not left_right else [i for i in range(386, 551)]+[j for j in range(386+550,551+550)]
            for video in self.video_list:
                for i in range(int(len(os.listdir(os.path.join(videos_path,str(video).zfill(4),"images")))/num_frames/self.frame_modulo)):
                    self.video_list_modified.append([video,frame_modulo*num_frames*i+1])

        # self.video_list.sort()
        
    def load_sequence(self, video_path, start_idx, key,mirror=False):
        seq = []
        for i in range(start_idx,start_idx+self.num_frames*self.frame_modulo,self.frame_modulo):
            img_path = os.path.join(video_path,key,str(i).zfill(4)+".jpg")
            if key == "maps":
                # open with gray scale
                img = Image.open(img_path).convert('L')
            else:
                img = Image.open(img_path)

            if mirror:
                #mirror img right-left if dataset is LR
                img=img.transpose(Image.FLIP_LEFT_RIGHT)
            if self.target_size is not None:
                img=img.resize((self.target_size[1],self.target_size[0]))
            img = np.array(img)
            img = torch.from_numpy(img)
            img = img.float()
            img = img/255
            if key=="maps":
                img = utils.prob_tensor(img)
            seq.append(img)
            # print(img_path)
        seq = torch.stack(seq, dim=0)
        return seq

    def revert_normalize(self, tensor):
        tensor = utils.revert_tensor_normalize(
            tensor,
            self._mean,
            self._std,
        )
        return tensor
    
    def load_specific_sq(self, video_num, start_idx):
        video_path = os.path.join(self.videos_path, str(video_num).zfill(4))
        frames = self.load_sequence(video_path, start_idx, "images")
        sals = self.load_sequence(video_path, start_idx, "maps")
        fixs = self.load_sequence(video_path, start_idx, "fixation")
        # normalize frames
        frames = utils.tensor_normalize(
            frames,
            self._mean,
            self._std,
        )

        # normalize sals
        # sals = utils.prob_tensor(sals)

        # binarize fix map
        fixs = torch.gt(fixs, 0.4)

        # permute to [T, C, H, W]
        frames=frames.permute(0,3,1,2).float()
        return frames, sals, fixs # return the fake label to use the same engine with kinetics dataset


    def __len__(self):
        return len(self.video_list_modified)

    def __getitem__(self, idx):
        '''
        Given the video idx, return the video frames
        num_frames frames are randomly selected from the videos
        Args:
            idx (int): the video index. 
        Returns:
            frames (tensor): [B, C, T, H, W]
            sals (tensor): [B, T, H, W] 
            fixs (tensor): [B, T, H, W]
        '''
        
        if self.mode == "train" or self.mode == "val":
            video_path = os.path.join(self.videos_path, str(self.video_list_modified[idx]).zfill(4))
            frame_list = os.listdir(os.path.join(self.videos_path, str(self.video_list_modified[idx]).zfill(4), "images"))
            # frame_list.sort()
            start_idx = np.random.randint(1,len(frame_list)-self.num_frames*self.frame_modulo+2) # random select the start frame
            frames = self.load_sequence(video_path, start_idx, "images",mirror= (self.left_right and self.video_list_modified[idx]>550))
            sals = self.load_sequence(video_path, start_idx, "maps",mirror= (self.left_right and self.video_list_modified[idx]>550))
            fixs = self.load_sequence(video_path, start_idx, "fixation",mirror= (self.left_right and self.video_list_modified[idx]>550))

        elif self.mode == "test":
            video_path = os.path.join(self.videos_path, str(self.video_list_modified[idx][0]).zfill(4))
            start_idx = self.video_list_modified[idx][1]
            frames = self.load_sequence(video_path, start_idx, "images",mirror= (self.left_right and self.video_list_modified[idx][0]>550))
            sals = self.load_sequence(video_path, start_idx, "maps",mirror= (self.left_right and self.video_list_modified[idx][0]>550))
            fixs = self.load_sequence(video_path, start_idx, "fixation",mirror= (self.left_right and self.video_list_modified[idx][0]>550))

        # load frames
        # frames = self.load_sequence(video_path, start_idx, "images",mirror= (self.left_right and self.video_list_modified[idx]>550))
        # sals = self.load_sequence(video_path, start_idx, "maps",mirror= (self.left_right and self.video_list_modified[idx]>550))
        # fixs = self.load_sequence(video_path, start_idx, "fixation",mirror= (self.left_right and self.video_list_modified[idx]>550))

        # normalize frames
        frames = utils.tensor_normalize(
            frames,
            self._mean,
            self._std,
        )

        # normalize sals
        # sals = utils.prob_tensor(sals)

        # binarize fix map
        fixs = torch.gt(fixs, 0.4)

        # permute to [T, C, H, W]
        frames=frames.permute(0,3,1,2).float()
        if self.mode=="test":
            video_num = self.video_list_modified[idx][0]
            start_idx = self.video_list_modified[idx][1]
            # to tensor
            video_num = torch.tensor(video_num)
            start_idx = torch.tensor(start_idx)
            return frames, sals, fixs, video_num, start_idx
        return frames, sals, fixs # return the fake label to use the same engine with kinetics dataset