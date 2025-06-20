import torch.nn as nn
from einops import rearrange
import math
from mmcv.cnn.bricks import ConvModule, build_activation_layer, build_norm_layer
from transxnet.MultiScaleDWConv import MultiScaleDWConv
class Mlp(nn.Module):  ### MS-FFN
    """
    Mlp implemented by with 1x1 convolutions.

    Input: Tensor with shape [B, C, H, W].
    Output: Tensor with shape [B, C, H, W].
    Args:
        in_features (int): Dimension of input features.
        hidden_features (int): Dimension of hidden features.
        out_features (int): Dimension of output features.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        drop (float): Dropout rate. Defaults to 0.0.
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_cfg=dict(type='GELU'),
                 drop=0,scale=(1, 3, 5, 7)):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=False),
            build_activation_layer(act_cfg),
            nn.BatchNorm2d(hidden_features),
        )
        self.dwconv = MultiScaleDWConv(hidden_features,scale=scale)
        self.act = build_activation_layer(act_cfg)
        self.norm = nn.BatchNorm2d(hidden_features)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_features),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # B,L,C=x.shape
        # h=w=int(math.sqrt(L))

        # x=rearrange(x,"b (h w) c -> b c h w",h=h,w=w)
        # print(x.shape)
        
        x = self.fc1(x)

        x = self.dwconv(x) + x
        x = self.norm(self.act(x))
        
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        # print(x.shape)
        # x=rearrange(x,"b c h w -> b (h w) c ")

        return x