from typing import Tuple
import numpy as np

from basic import hyp, names_lib
from utils import Assert
from .base import Dataset
from ..cropper import SimpleCropper, ScalingSimpleCropper


class ValidDataset(Dataset):
    def __init__(self, **kwargs):
        # 解析参数
        p = {
            # 应当由参数传递
            'name_list': names_lib['valid'],
            # 应当默认
            'use_format': True,
            # 应当由配置文件给定
            'use_migrate': hyp['dataset.use_migrate'],
            'use_confidence': hyp['dataset.use_confidence'],
            'use_scale': hyp['dataset.valid.use_scale'],
        }
        p.update(**kwargs)
        Assert.not_none(**p)

        if p['use_scale']:
            cropper = ScalingSimpleCropper(hyp['dataset.valid.scaling'], p['use_migrate'], p['use_confidence'])
        else:
            cropper = SimpleCropper(p['use_migrate'], p['use_confidence'])
        super().__init__(cropper, p['name_list'], use_format=p['use_format'])

    def __getitem__(self, item: int) -> Tuple[np.ndarray, np.ndarray, str]:
        grid_info = self.samples[item]
        return *self.get(grid_info), grid_info['name']
