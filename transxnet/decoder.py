import torch.nn as nn
import torch
from transxnet.Mlp import Mlp as PatchModify_transXNet
from pytools.PatchExpend import PatchExpend
import math
from einops import rearrange
class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        #print(x.shape)
        x = self.conv(x)
        return x
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.in_planes = in_planes
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool_out = self.avg_pool(x) 
        avg_out = self.fc2(self.relu1(self.fc1(avg_pool_out)))
        #print(x.shape)
        max_pool_out= self.max_pool(x) #torch.topk(x,3, dim=1).values

        max_out = self.fc2(self.relu1(self.fc1(max_pool_out)))
        out = avg_out + max_out
        return self.sigmoid(out) 
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x) 

class transDecoder1(nn.Module):
    def __init__(self, embed_dim: list, patch_size: int,) -> None:
        super().__init__()
        self.CA4 = ChannelAttention(embed_dim[0])
        self.CA3 = ChannelAttention(embed_dim[1])
        self.CA2 = ChannelAttention(embed_dim[2])
        self.CA1 = ChannelAttention(embed_dim[3])
        self.Conv_1x1 = nn.Conv2d(embed_dim[0],embed_dim[0],kernel_size=1,stride=1,padding=0)
        
        self.SA = SpatialAttention()
        self.ConvBlock4 = conv_block(ch_in=embed_dim[0], ch_out=embed_dim[0])
        # up stage4#######################################################################################################
        # self.up_stage4 = None

        self.up_expend4 = PatchExpend(
            dim=embed_dim[0], out_channels=embed_dim[1])
        # self.up_norm4 = nn.BatchNorm2d(embed_dim[0])
        # up stage4#######################################################################################################
        # up stage3#######################################################################################################

        self.up_concat_linear3 = PatchModify_transXNet(512, 512*2, 256)
        self.ConvBlock3 = conv_block(ch_in=embed_dim[1], ch_out=embed_dim[1])
        self.up_expend3 = PatchExpend(
            dim=embed_dim[1], out_channels=embed_dim[2])
        self.up_norm3 = nn.BatchNorm2d(embed_dim[1])

        # up stage3#######################################################################################################
        # up stage2#######################################################################################################

        self.up_concat_linear2 = PatchModify_transXNet(256, 256*2, 128)
        self.ConvBlock2 = conv_block(ch_in=embed_dim[2], ch_out=embed_dim[2])
        self.up_expend2 = PatchExpend(
            dim=embed_dim[2], out_channels=embed_dim[3])
        self.up_norm2 = nn.BatchNorm2d(embed_dim[2])

        # up stage2#######################################################################################################
        # up stage1#######################################################################################################
        self.up_concat_linear1 = PatchModify_transXNet(128, 128*2, 64)
        self.up_norm1 = nn.BatchNorm2d(embed_dim[3])
        self.ConvBlock1 = conv_block(ch_in=embed_dim[3], ch_out=embed_dim[3])
        # up stage1#######################################################################################################
        # output head#####################################################################################################

        
        # output head#####################################################################################################

    def forward(self, x, x_list):
        
        
        # up stage4##################################################################################################
        d4 = self.Conv_1x1(x)
        d4 = self.CA4(d4)*d4
        d4 = self.SA(d4)*d4 
        d4 = self.ConvBlock4(d4)
        d3 = self.up_expend4(d4)

        # up stage4##################################################################################################
        # up stage3##################################################################################################
        
        d3 = torch.cat([d3, x_list[0]], 1)
        d3 = self.up_concat_linear3(d3)
        d3 = self.up_norm3(d3)
        d3 = self.CA3(d3)*d3
        d3 = self.SA(d3)*d3        
        d3 = self.ConvBlock3(d3)
        d2 = self.up_expend3(d3)
        # up stage3##################################################################################################
        # up stage2##################################################################################################
        d2 = torch.cat([d2, x_list[1]], 1)
        d2 = self.up_concat_linear2(d2)
        d2 = self.up_norm2(d2)
        d2 = self.CA2(d2)*d2
        d2 = self.SA(d2)*d2
        d2 = self.ConvBlock2(d2)
        d1 = self.up_expend2(d2)
        # up stage2#################################################################################################
        # up stage1##################################################################################################
        d1 = torch.cat([d1, x_list[2]], 1)
        d1 = self.up_concat_linear1(d1)

        d1 = self.up_norm1(d1)
        d1 = self.CA1(d1)*d1
        d1 = self.SA(d1)*d1
        d1 = self.ConvBlock1(d1)
        # up stage1##################################################################################################
        return d4, d3, d2, d1
