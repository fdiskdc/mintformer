import torch
# from torchvision.models import resnet18
from thop import profile
from torchinfo import summary
from functools import partial
import torch.nn as nn


from lib.networks import MaxViT, MaxViT4Out, MaxViT_CASCADE, MERIT_Parallel, MERIT_Cascaded
from functools import partial


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


model = MERIT_Cascaded(n_class=9, img_size_s1=(256, 256), img_size_s2=(
    224, 224), model_scale='small', decoder_aggregation='additive', interpolation='bilinear')

# model = MERIT_Cascaded(
#     img_size=224,
#     in_chans=1,
#     embed_dim=[64, 128, 256, 512],
#     depth=[2, 2, 2, 2],
#     split_size=[1, 2, 7, 7], num_heads=[4, 8, 16, 32], mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
#     drop_path_rate=0., num_classes=9
# )
#
# arr1=torch.randn(24,3,224,224)
# flops, params = profile(model, inputs=(arr1, ))
# print('flops1:{}'.format(flops))
# print('params1:{}'.format(params))
# torch.rand((1,1,224,224))
# inputs = torch.zeros((1,1,224,224), dtype=torch.long)
summary(model, input_size=(24, 1, 256, 256))
# print('parameters_count:',count_parameters(model))
# pyfloats
# summary的print设置为true，输出一下，对着输出每一层输出查看
