from typing import Tuple, Dict, Any
import numpy as np
from torch.utils.data.dataloader import Dataset as StandardDataset

from basic import names_lib, hyp
from utils import Assert, Timer
from .base import Dataset
from ..cropper import GroupCropper, ScalingGroupCropper
from ..formatter import weights_formatter


class TestDataSet(Dataset):
    def __init__(self, T: Timer = None, **kwargs):
        # 解析参数
        p = {
            # 应当由参数传递
            'name_list': names_lib['test'],
            # 应当默认
            'use_format': True,
            # 应当由配置文件给定
            'use_migrate': hyp['dataset.use_migrate'],
            'use_confidence': hyp['dataset.use_confidence'],
            'use_box': hyp['dataset.test.use_box'],
            'use_scale': hyp['dataset.test.use_scale'],
            'scaling': hyp['dataset.test.scaling'],
        }
        p.update(**kwargs)
        Assert.not_none(**p)

        if p['use_scale']:
            cropper = ScalingGroupCropper(
                scaling=p['scaling'],
                use_migrate=p['use_migrate'],
                use_confidence=p['use_confidence'],
                use_box=p['use_box'],
            )
        else:
            # GroupCropper 尚未实现 use_box=False
            cropper = GroupCropper(p['use_migrate'], p['use_confidence'])

        super().__init__(cropper=cropper, names_list=p['name_list'], use_format=p['use_format'], T=T)

    def __getitem__(self, item: int) -> Exception:
        raise NotImplemented('Never predict in datasource-loader')

    def __iter__(self) -> iter:
        for group in self.samples:
            yield Subset(group_info=group, pset=self)

    def pget(self, grid: dict) -> np.ndarray:
        image = self.cropper.get(**grid)
        if self.use_format:
            return weights_formatter.image(image)
        else:
            return image

    def plab(self, grid: dict) -> np.ndarray:
        label = self.cropper.lab(**grid)
        if self.use_format:
            return weights_formatter.label(label)
        else:
            return label

    def get_image(self, name: str):
        return self.cropper.get_image(name)

    def get_label(self, name: str):
        label = self.cropper.get_label(name)
        assert label is not None, 'Not ready for full-image-predict!'
        return label


class Subset(StandardDataset):
    def __init__(self, group_info: dict, pset: Dataset):
        self.name = group_info['name']
        self.patches = []
        patch_size = hyp['cropper.patch_size']
        for s in group_info['scaling']:
            scale = s['scale']
            size = patch_size * scale
            size = round(size)
            for g in s['grids']:
                self.patches.append({
                    'name': self.name,
                    'scale': scale,
                    'size': size,
                    'x': g['x'],
                    'y': g['y'],
                })
        self.pset = pset

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, index) -> Tuple[Dict[str, Any], np.ndarray]:
        grid = self.patches[index]
        return grid, self.pset.pget(grid)

    def get_HW(self) -> Tuple[int, int]:
        return self.pset.get_HW(self.name)

    def get_image(self) -> np.ndarray:
        return self.pset.get_image(self.name)

    def get_label(self) -> np.ndarray:
        return self.pset.get_label(self.name)

    def get_box(self) -> Tuple[int, int, int, int]:
        return self.pset.get_box(self.name)
