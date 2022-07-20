import math

from basic import config
from utils import Rotator
from .interface import Cropper
from ..adaptor import Adaptor
from ..loader import Loader


# 均匀切图
class SimpleCropper(Cropper):
    def __init__(self, adaptor: Adaptor):
        self.adaptor = adaptor
        self.patch_size = config['cropper.patch_size']
        self.step = config['cropper.step']

    def sample(self, loader: Loader, num: int):
        # 理想区域
        l, u, r, d = self.adaptor.get_box(loader)
        # 采样所需的角度、缩放信息
        degrees = self.adaptor.get_rotate_degrees()
        scalings = self.adaptor.get_scaling_rates()
        result = []
        for degree in degrees:
            # 采样变换后的窗口大小、步长
            rotate_size_rate = Rotator.size_rate(degree)
            for scaling in scalings:
                size = math.ceil(self.patch_size * scaling * rotate_size_rate)
                step = math.floor(self.step * scaling)
                # 简单遍历(不足遍历)
                for i, x in enumerate(range(l, r-size+step, self.step)):
                    for j, y in enumerate(range(u, d-size+step, self.step)):
                        result.append({
                            'part_id': loader.part_id,
                            'name': loader.name,
                            'i': i,
                            'j': j,
                            # (x, y) 记录中点坐标
                            'x': x + size//2,
                            'y': y + size//2,
                            'w': self.patch_size,
                            'h': self.patch_size,
                            'degree': degree,
                            'scaling': scaling,
                        })
        if num != -1: return result[:num]
        else: return result
