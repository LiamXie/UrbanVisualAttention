import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from util.convlstm import ConvLSTM
from util import utils

class UEN(nn.Module):
    def __init__(self,smoothing_kernel_size=41
                 ):
        super(UEN, self).__init__()

        # cnn backbone\
        self.cnn=models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.cnn.fc = torch.nn.Identity()
        self.cnn.avgpool = torch.nn.Identity()
        self.cnn_out_channels=512
        self.cnn_2x_channels=256
        self.cnn_4x_channels=128

        # rnn
        self.rnn=ConvLSTM(
                    input_dim=self.cnn_out_channels,
                    hidden_dim=self.cnn_out_channels,
                    kernel_size=(3,3),
                    num_layers=1,
                    batch_first=True,
                    bias=False,
                    return_all_layers=False
                    )
        
        # skip connection
        self.skip_2x_channels=int(self.cnn_2x_channels/2)
        self.skip_2x=nn.Sequential(
            nn.Conv2d(self.cnn_2x_channels, self.skip_2x_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.skip_2x_channels),
            nn.ReLU(inplace=True)
        )
        self.skip_4x_channels=int(self.cnn_4x_channels/2)
        self.skip_4x=nn.Sequential(
            nn.Conv2d(self.cnn_4x_channels, self.skip_4x_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.skip_4x_channels),
            nn.ReLU(inplace=True)
        )
        
        # upsample
        self.upsample_1=nn.Sequential(
            nn.ConvTranspose2d(self.cnn_out_channels, self.cnn_2x_channels, kernel_size=3, stride=2, padding=1, output_padding=(0,1)),
        )
        self.upsample_2=nn.Sequential(
            nn.ConvTranspose2d(self.cnn_2x_channels+self.skip_2x_channels, self.cnn_4x_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(self.cnn_4x_channels),
            nn.ReLU(inplace=True)
        )
        self.upsample_3=nn.Sequential(
            nn.ConvTranspose2d(self.cnn_4x_channels+self.skip_4x_channels, self.cnn_4x_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(self.cnn_4x_channels),
            nn.ReLU(inplace=True)
        )
        self.upsample_4=nn.Sequential(
            nn.ConvTranspose2d(self.cnn_4x_channels, self.cnn_4x_channels, kernel_size=3, stride=2, padding=1, output_padding=(0,1)),
            nn.BatchNorm2d(self.cnn_4x_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.cnn_4x_channels, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

        self.smoothing_kernel_size = smoothing_kernel_size
        self.smoothing = nn.Conv2d(1, 1, kernel_size=self.smoothing_kernel_size, padding=0, bias=False)
        # self.smoothing.weight.data.nor


    def _forward_cnn(self, x):
        feat_2x, feat_4x= None, None 
        x = self.cnn.conv1(x)
        x = self.cnn.bn1(x)
        x = self.cnn.relu(x)
        x = self.cnn.maxpool(x)

        x = self.cnn.layer1(x)
        x = self.cnn.layer2(x)
        feat_4x=x.clone()
        x = self.cnn.layer3(x)
        feat_2x=x.clone()
        x = self.cnn.layer4(x)

        return x, feat_2x, feat_4x
    
    def forward(self, x):
        feat_2x_list, feat_4x_list = [], []
        for t, img in enumerate(torch.unbind(x, dim=1)):
            feat, feat_2x, feat_4x = self._forward_cnn(img)
            if t == 0:
                output = torch.zeros(x.shape[0], x.shape[1], feat.shape[1], feat.shape[2], feat.shape[3]).to(x.device)
            output[:, t, :, :, :] = feat
            # feat_list.append(feat)
            feat_2x_list.append(feat_2x)
            feat_4x_list.append(feat_4x)
        
        output, _ = self.rnn(output)

        a=list(zip(torch.unbind(output,dim=1),feat_2x_list, feat_4x_list))
        for t, (feat, feat_2x, feat_4x) in enumerate(list(zip(torch.unbind(output,dim=1),feat_2x_list, feat_4x_list))):
            feat_up_2x=self.upsample_1(feat)
            feat_up_2x=torch.cat((feat_up_2x, self.skip_2x(feat_2x)), dim=1)
            feat_up_4x=self.upsample_2(feat_up_2x)
            feat_up_4x=torch.cat((feat_up_4x, self.skip_4x(feat_4x)), dim=1)
            feat_up=self.upsample_3(feat_up_4x)
            feat_up=self.upsample_4(feat_up)
            # feat_up=feat_up.squeeze(1)
            if t == 0:
                pred = torch.zeros(x.shape[0], x.shape[1], feat_up.shape[-2], feat_up.shape[-1]).to(x.device)

            #smooth
            feat_up = F.pad(feat_up, [self.smoothing_kernel_size // 2] * 4,
                            mode='replicate')
            
            feat_up = self.smoothing(feat_up)
            feat_up = feat_up.squeeze(1)
            
            feat_up = utils.log_softmax(feat_up) # output is log-probability
            pred[:, t, :, :, ] = feat_up
        return pred

if __name__=="__main__":
    net=UEN()