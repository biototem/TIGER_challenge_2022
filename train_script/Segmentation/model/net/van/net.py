import torch
from typing import TypeVar
import functools
import timm
from segmentation_models_pytorch.base.modules import Activation
import torch.nn.functional as F

from .impl_van import VAN
from .impl_uper import LikeUPerHead
from .impl_fcn import FCNHead


T = TypeVar('T', bound=torch.nn.Module)

VAN_ARGS = {
    'img_size': 512,
    'patch_size': 4,
    'in_chans': 3,
    'num_classes': 1000,
    'embed_dims': [64, 128, 320, 512],
    'num_heads': [1, 2, 5, 8],
    'mlp_ratios': [8, 8, 4, 4],
    'qkv_bias': True,
    'qk_scale': None,
    'drop_rate': .0,
    'attn_drop_rate': 0.0,
    'drop_path_rate': 0.1,
    'norm_layer': functools.partial(torch.nn.LayerNorm, eps=1e-06),
    'depths': [3, 3, 12, 3],
    'sr_ratios': [8, 4, 2, 1],
    'num_stages': 4,
    'linear': False,
}

UPER_ARGS = {
    'in_dims': [64, 128, 320, 512],
    'fc_dim': 4096,
    'fpn_dim': 256,
    'out_dim': 8,
    # 'pool_scales': [1, 2, 3, 6],
    'pool_scales': [1, 1, 2, 3],
}

FCN_ARGS = {
    'num_convs': 2,
    'kernel_size': 3,
    'concat_input': True,
    'dilation': 1,
    'in_channels': 320,
    'in_index': 2,
    'channels': 256,
    'dropout_ratio': 0.1,
    'num_classes': 8,
    'norm_cfg': {'type': 'BN', 'requires_grad': False},
    'align_corners': False,
    'loss_decode': {
        'type': 'CrossEntropyLoss',
        'use_sigmoid': False,
        'loss_weight': 0.4
    }
}


class Net(torch.nn.Module):
    model_id = 'van'

    def __init__(self, in_dim: int, out_dim: int, activation: torch.nn.Module = None, pretrained: str = None, use_auxiliary: bool = True):
        super().__init__()
        self.encoder = VAN(**{**VAN_ARGS, 'in_chans': in_dim})
        if pretrained:
            self.encoder.init_weights(pretrained=pretrained)

        self.decoder = LikeUPerHead(**{**UPER_ARGS, 'out_dim': out_dim})
        self.auxiliary = use_auxiliary and FCNHead(**{**FCN_ARGS, 'num_classes': out_dim})
        self.activation = activation and Activation(activation)

        self.use_auxiliary = use_auxiliary

    def forward(self, image: torch.tensor):
        h, w = image.shape[2:]
        features = self.encoder(image)
        # print([f.shape for f in features])
        # 正常结果
        seg_result = self.decoder(features)
        seg_result = F.interpolate(seg_result, size=(h, w), mode='bilinear', align_corners=False)
        if self.activation:
            seg_result = self.activation(seg_result)
        # 当启用辅助网络、且处于训练状态时，计算和返回辅助网络结果
        if self.use_auxiliary and self.training:
            aux_result = self.auxiliary(features)
            aux_result = F.interpolate(aux_result, size=(h, w), mode='bilinear', align_corners=False)
            if self.activation:
                aux_result = self.activation(aux_result)
            return [seg_result, aux_result]
        # 否则返回单一结果
        return seg_result
