from typing import List, Tuple, Any, Union
import numpy as np

from utils import Shape, one_hot_numpy
from .rotate import trans_rotate, get_rotate_degrees
from .scaling import trans_scaling, get_scaling_rates
from .transfer import trans_transfer
from .formatter import trans_formatter
from ..loader import Loader


class Adaptor(dict):
    def __init__(self, **kwargs):
        super().__init__()
        self.update(kwargs)
        # 根据约定格式解析旋转配置
        self.config_rotate = trans_rotate(self['cropper_rotate'])
        # 根据约定格式解析缩放配置
        self.config_scaling = trans_scaling(self['cropper_scaling'])
        # 根据约定格式生成数据增强器
        self.transfer_impl = trans_transfer(self['transfer'])
        # 根据约定格式生成数据规范器
        self.formatter_impl = trans_formatter(self['formatter'])

    # 根据业务需求，给出旋转角度：固定值或随机值
    def get_rotate_degrees(self) -> List[float]:
        return get_rotate_degrees(**self.config_rotate)

    # 根据业务需求，给出缩放比例：固定值或随机值
    def get_scaling_rates(self) -> List[float]:
        return get_scaling_rates(**self.config_scaling)

    # 根据业务需求和具体数据，确定采样的坐标范围
    def get_sight(self, loader: Loader) -> Tuple[int, int, int, int]:
        w, h = loader.size
        return loader.sight or (0, 0, w, h)

    # 根据业务需求和具体数据，确定采样的理想范围
    def get_box(self, loader: Loader) -> Tuple[int, int, int, int]:
        w, h = loader.size
        return loader.box or self.get_sight(loader)

    def get_margin(self, loader: Loader) -> Tuple[int, int, int, int]:
        l, u, r, d = self.get_box(loader)
        margin = self['cropper_margin']
        L, U, R, D = self.get_sight(loader)
        ml = min(l - L, margin)
        mu = min(u - U, margin)
        mr = min(R - r, margin)
        md = min(D - d, margin)
        return ml, mu, mr, md

    # 根据业务需求，代理返回值格式
    def get_returns(self, grid, loader: Loader) -> dict:
        returns = {
            'image': loader.image(grid),    # 总是返回图像
            'label': None,                  # 根据需求返回标签
            'grid': None,                     # 根据需求返回坐标
        }
        if self['return_label']: returns['label'] = loader.label(grid)
        if self['return_grid']: returns['grid'] = grid
        return returns

    # 执行数据增强
    def transfer(self, image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if label is not None:
            result = self.transfer_impl(image=image, mask=label.astype(np.uint8))
            image, label = result['image'], result['mask']
        else:
            result = self.transfer_impl(image=image)
            image, label = result['image'], None
        return image, label

    # 执行数据标准化 - 图片标准化
    def format_image(self, image: np.ndarray) -> np.ndarray:
        return self.formatter_impl.image(image)

    # 执行数据标准化 - 标签标准化
    def format_label(self, label: np.ndarray) -> np.ndarray:
        if label is None: return None
        # 基于比赛需求：
        if len(label.shape) == 2:
            label = one_hot_numpy(label, num=self['class_num'])
        label[label.sum(axis=2) == 0, 0] = 1
        return self.formatter_impl.label(label)
