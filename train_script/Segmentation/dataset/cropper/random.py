import math
import numpy as np

from basic import config
from utils import Rotator
from .interface import Cropper
from ..adaptor import Adaptor
from ..loader import Loader


# 随机切图
class RandomCropper(Cropper):
    def __init__(self, adaptor: Adaptor):
        self.adaptor = adaptor
        self.patch_size = config['cropper.patch_size']

    def sample(self, loader: Loader, num: int = -1):
        # 理想区域
        l, u, r, d = self.adaptor.get_box(loader)
        # 松弛距离
        ml, mu, mr, md = self.adaptor.get_margin(loader)
        result = []
        # 随机遍历(定数量遍历)
        if num == -1:
            area = (r - l) * (d - u)
            num = math.ceil(area / self.patch_size ** 2)
        while len(result) < num:
            # 采样所需的角度、缩放信息
            degrees = self.adaptor.get_rotate_degrees()
            scalings = self.adaptor.get_scaling_rates()
            for degree in degrees:
                # 采样变换后的窗口大小、步长
                rotate_size_rate = Rotator.size_rate(degree)
                for scaling in scalings:
                    size = math.ceil(self.patch_size * scaling * rotate_size_rate)
                    # 随机边界：若 margin > 0 采样区域不超过 margin；否则不超过 box
                    rl = max(l, l+ml) + size // 2
                    ru = max(u, u+mu) + size // 2
                    rr = min(r, r-mr) - size // 2
                    rd = min(d, d-md) - size // 2
                    if rr <= rl or rd <= ru: continue
                    x = np.random.randint(rl, rr)
                    y = np.random.randint(ru, rd)
                    result.append({
                        'part_id': loader.part_id,
                        'name': loader.name,
                        'i': -1,
                        'j': -1,
                        # (x, y) 记录中点坐标
                        'x': x,
                        'y': y,
                        'w': self.patch_size,
                        'h': self.patch_size,
                        'degree': degree,
                        'scaling': scaling,
                    })
        return result[:num]
