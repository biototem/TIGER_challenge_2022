import sys
import os
sys.path.append(os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )


def ConvBnAct(in_ch, out_ch, ker_sz, stride, pad, act=nn.Identity(), group=1, dilation=1):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, ker_sz, stride, pad, groups=group, bias=False, dilation=dilation),
                         nn.BatchNorm2d(out_ch, eps=1e-8, momentum=0.9),
                         act)


class LikeUPerHead(nn.Module):
    def __init__(self, in_dims=(128, 256, 512), fc_dim=4096, fpn_dim=256, out_dim=3, pool_scales=(2, 4, 6)):
        super().__init__()

        n_in = len(in_dims)
        
        act = nn.ReLU(True)

        # FC
        self.gpool_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnAct(in_dims[-1], fc_dim, 1, 1, 0, act),
        )

        # PPM Module
        self.ppm_pool = nn.ModuleList()
        self.ppm_conv = nn.ModuleList()

        for scale in pool_scales:
            self.ppm_pool.append(nn.Upsample(scale_factor=1 / scale, mode='bilinear', align_corners=False))
            self.ppm_conv.append(ConvBnAct(in_dims[-1], 512, 1, 1, 0, act))

        self.ppm_last_conv = ConvBnAct(fc_dim + in_dims[-1] + len(pool_scales)*512, fpn_dim, 1, 1, 0, act)

        # FPN Module
        self.fpn_in = nn.ModuleList()
        for fpn_in_ch in in_dims[:-1]:  # skip the top layer
            self.fpn_in.append(ConvBnAct(fpn_in_ch, fpn_dim, 1, 1, 0, act))

        self.fpn_out = nn.ModuleList()
        for _ in range(n_in - 1):       # skip the top layer
            self.fpn_out.append(ConvBnAct(fpn_dim, fpn_dim, 1, 1, 0, act))

        self.conv_fusion = ConvBnAct(fpn_dim * n_in, fpn_dim, 1, 1, 0, act)

        self.out_conv = nn.Sequential(
            ConvBnAct(fpn_dim, fpn_dim, 3, 1, 1, act),
            nn.Conv2d(fpn_dim, out_dim, 1, 1, 0, bias=False)
        )

    def forward(self, xs):

        y_end = xs[-1]
        y_end_hw = y_end.shape[2:]

        y_fc = self.gpool_fc(y_end)
        y_fc = F.interpolate(y_fc, y_end_hw, mode='bilinear', align_corners=False)

        ppm_out = [y_fc, y_end]
        for i in range(len(self.ppm_pool)):
            y = self.ppm_pool[i](y_end)
            y = self.ppm_conv[i](y)
            y = F.interpolate(y, y_end_hw, mode='bilinear', align_corners=False)
            ppm_out.append(y)

        ppm_out = torch.cat(ppm_out, 1)
        ppm_out = self.ppm_last_conv(ppm_out)

        y = ppm_out
        del ppm_out

        fpn_feature_list = [y]

        # 注意，这里的最后一层是不参与的
        for i in reversed(range(len(self.fpn_in))):
            y_l = xs[i]
            y_l = self.fpn_in[i](y_l)
            y = F.interpolate(y, size=y_l.shape[2:], mode='bilinear', align_corners=False)
            y = y + y_l
            y = self.fpn_out[i](y)
            fpn_feature_list.append(y)

        fpn_feature_list = fpn_feature_list[::-1]
        del y

        # 把特征层上采样到最后
        out_hw = fpn_feature_list[0].shape[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            y = F.interpolate(fpn_feature_list[i], out_hw, mode='bilinear', align_corners=False)
            fusion_list.append(y)
        fusion_out = torch.cat(fusion_list, 1)
        y = self.conv_fusion(fusion_out)

        y = self.out_conv(y)
        return y


if __name__ == '__main__':
    net = LikeUPerHead([128, 256, 512], 4096, 256, 3, [2, 4, 6])
    a = torch.zeros([1, 128, 128, 128])
    b = torch.zeros([1, 256, 64, 64])
    c = torch.zeros([1, 512, 32, 32])
    net.eval()

    o = net([a,b,c])

    print(o)

