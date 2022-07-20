import math
import numpy as np

from basic import config
from utils import magic_iter, Rotator
from .interface import Cropper
from ..adaptor import Adaptor
from ..loader import Loader


# 密集切图
class DenseCropper(Cropper):
    def __init__(self, adaptor: Adaptor):
        self.adaptor = adaptor
        self.patch_size = config['cropper.patch_size']
        self.step = config['cropper.step']

    def sample(self, loader: Loader, num: int):
        # 理想区域
        l, u, r, d = self.adaptor.get_box(loader)
        # 松弛距离
        ml, mu, mr, md = self.adaptor.get_margin(loader)
        # 采样所需的角度、缩放信息
        degrees = self.adaptor.get_rotate_degrees()
        scalings = self.adaptor.get_scaling_rates()
        result = []
        for degree in degrees:
            # 生成旋转密叠信息
            radius = degree * np.pi / 180
            sina = np.sin(radius)
            cosa = np.cos(radius)
            seca = 1 / cosa
            px = self.step * seca
            qx = self.step * sina
            qy = self.step * cosa
            # 采样变换后的窗口大小、步长
            rotate_size_rate = Rotator.size_rate(degree)
            for scaling in scalings:
                size = math.ceil(self.patch_size * scaling * rotate_size_rate)
                # 密叠遍历 -> 松弛距离大于0表示超越边界、小于0表示低于边界
                anchor = l + np.sign(ml) * np.random.randint(abs(ml) + 1)
                for j, y in magic_iter(
                        target=d - u,
                        unit=size,
                        step=qy,
                        # margin=(box[1], sight[3] - box[3])
                        margin=(mu, md)
                ):
                    x = anchor
                    # 需保证 x、y 均在允差之内
                    while x < min(0, -ml):
                        x += px
                    i = 0
                    while x + size < max(r-l, r-l+mr):
                        result.append({
                            'part_id': loader.part_id,
                            'name': loader.name,
                            'i': i,
                            'j': j,
                            'x': l + x + size//2,
                            'y': u + y + size//2,
                            'w': self.patch_size,
                            'h': self.patch_size,
                            'degree': degree,
                            'scaling': scaling,
                        })
                        x += px
                        i += 1
                    # qx 可正可负，但锚点永远固定在行首
                    anchor += qx
                    if anchor >= min(0, -ml) + px:
                        anchor -= px
                    elif anchor < min(0, -ml):
                        anchor += px
        if num != -1: return result[:num]
        else: return result
