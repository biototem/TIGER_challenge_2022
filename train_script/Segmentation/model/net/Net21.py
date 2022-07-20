'''
EfficientNetV2_m+LikeUPerNet
'''

import sys
import os
sys.path.append(os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from likeuperhead import LikeUPerHead
# from model_utils_torch import resize_ref


@torch.jit.script
def resize_ref(x, shortpoint, method: str = 'bilinear', align_corners: bool = None):
    """
    :type x: torch.Tensor
    :type shortpoint: torch.Tensor
    :type method: str
    :type align_corners: bool
    """
    hw = shortpoint.shape[2:4]
    ihw = x.shape[2:4]
    if hw != ihw:
        x = torch.nn.functional.interpolate(x, hw, mode=method, align_corners=align_corners)
    return x


def ConvBnAct(in_ch, out_ch, ker_sz, stride, pad, act=nn.Identity(), group=1, dilation=1):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, ker_sz, stride, pad, groups=group, bias=False, dilation=dilation),
                         nn.BatchNorm2d(out_ch, eps=1e-8, momentum=0.9),
                         act)


class UpSampleConcatConvLayer(nn.Module):
    def __init__(self, in1_dim, in2_dim, out_dim):
        super().__init__()
        self.m1 = nn.Sequential(ConvBnAct(in1_dim+in2_dim, out_dim, 1, 1, 0, nn.SiLU()),
                                ConvBnAct(out_dim, out_dim, 3, 1, 1, nn.SiLU()))

    def forward(self, x1, x2):
        x1 = resize_ref(x1, x2)
        y = torch.cat([x1, x2], 1)
        y = self.m1(y)
        return y


class Net(nn.Module):
    model_id = 'net21'

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.m1 = timm.create_model('efficientnetv2_m', in_chans=in_dim, output_stride=32, features_only=True, pretrained=False)
        self.m2 = LikeUPerHead([48, 80, 176, 512], 4096, 256, out_dim, [2, 4, 6])

        # self.up1 = UpSampleConcatConvLayer(512, 176, 176)
        # self.up2 = UpSampleConcatConvLayer(176, 80, 80)
        # self.up3 = UpSampleConcatConvLayer(80, 48, 48)
        # self.up4 = UpSampleConcatConvLayer(48, 24, 32)
        # self.out1 = ConvBnAct(32, out_dim, 1, 1, 0)

    def forward(self, x):
        ys = self.m1(x)
        # print([tuple(k.shape) for k in ys])
        y = self.m2(ys[1:])
        y = F.interpolate(y, scale_factor=4., mode='bilinear', align_corners=False)

        # y = ys[4]
        # y = self.up1(y, ys[3])
        # y = self.up2(y, ys[2])
        # y = self.up3(y, ys[1])
        # y = self.up4(y, ys[0])
        # y = self.out1(y)
        # y = F.interpolate(y, scale_factor=2., mode='bilinear', align_corners=False)
        return y


if __name__ == '__main__':
    import model_utils_torch
    a = torch.zeros(2, 3, 512, 512).cuda(0)
    # a = torch.zeros(2, 512, 512, 3).cuda(0).permute(0, 3, 1, 2)
    net = Net(3, 3).cuda(0)
    model_utils_torch.print_params_size(net)
    y = net(a)
    y.sum().backward()
    print(y.shape)
