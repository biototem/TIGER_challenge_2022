"""
EfficientNetV2_s+LikeUPerNet
"""

import sys
import os

from model.net.van.impl_fcn import FCNHead
from model.net.van.net import FCN_ARGS

sys.path.append(os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from likeuperhead import LikeUPerHead
from segmentation_models_pytorch.base.modules import Activation
# from model_utils_torch import resize_ref


@torch.jit.script
def resize_ref(x, short_point, method: str = 'bilinear', align_corners: bool = None):
    """
    :type x: torch.Tensor
    :type short_point: torch.Tensor
    :type method: str
    :type align_corners: bool
    """
    hw = short_point.shape[2:4]
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
    model_id = 'net24'

    def __init__(self, in_dim, out_dim, activation=None, use_auxiliary: bool = False):
        super().__init__()
        self.m1 = timm.create_model('tf_efficientnetv2_s_in21k', in_chans=in_dim, output_stride=32, features_only=True, pretrained=True)
        self.m2 = LikeUPerHead([48, 64, 160, 256], 4096, 256, out_dim, [2, 4, 6])
        self.auxiliary = use_auxiliary and FCNHead(**{**FCN_ARGS, 'in_channels': 160, 'num_classes': out_dim})
        self.activation = activation and Activation(activation)

        self.use_auxiliary = use_auxiliary
        # self.up1 = UpSampleConcatConvLayer(512, 176, 176)
        # self.up2 = UpSampleConcatConvLayer(176, 80, 80)
        # self.up3 = UpSampleConcatConvLayer(80, 48, 48)
        # self.up4 = UpSampleConcatConvLayer(48, 24, 32)
        # self.out1 = ConvBnAct(32, out_dim, 1, 1, 0)

    def forward(self, x):
        ys = self.m1(x)
        # print([tuple(k.shape) for k in ys])
        # 正常结果
        y = self.m2(ys[1:])
        y = F.interpolate(y, scale_factor=4., mode='bilinear', align_corners=False)
        if self.activation:
            y = self.activation(y)

        # 当启用辅助网络、且处于训练状态时，计算和返回辅助网络结果
        if self.use_auxiliary and self.training:
            z = self.auxiliary(ys[1:])
            z = F.interpolate(z, scale_factor=16., mode='bilinear', align_corners=False)
            if self.activation:
                z = self.activation(z)
            return [y, z]
        # 否则返回单一结果
        return y


if __name__ == '__main__':
    import lib.model_utils_torch as mut
    a = torch.zeros(2, 3, 512, 512).cuda(0)
    # a = torch.zeros(2, 512, 512, 3).cuda(0).permute(0, 3, 1, 2)
    net = Net(3, 3).cuda(0)
    print(net._get_name())
    mut.print_params_size(net)
    y = net(a)
    y.sum().backward()
    print(y.shape)
    # print(net)
