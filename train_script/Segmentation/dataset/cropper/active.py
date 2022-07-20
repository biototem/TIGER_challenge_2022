from typing import List
import os
import json
import math
import bisect
import numpy as np

from basic import config, join
from utils import magic_iter, Rotator
from .interface import Cropper
from ..adaptor import Adaptor
from ..loader import Loader


# 随机切图
class ActiveCropper(Cropper):
    def __init__(self, adaptor: Adaptor):
        self.adaptor = adaptor
        self.patch_size = config['cropper.patch_size']
        self.weights = config['dataset.active_weight']
        self.cache = {}

    # 加载主动均衡采样预备数组
    def load(self, loader: Loader) -> List[int]:
        # 如果内存中有, 直接返回
        if loader.part_id in self.cache:
            return self.cache[loader.part_id]
        # 否则检查硬盘
        cache_path = join('~/cache/active_sample_weights', str(self.weights), f'{loader.part_id}.json')
        # 如果硬盘中有, 加载到内存并返回
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                result = json.load(f)
            self.cache[loader.part_id] = result
            return result
        # 否则就需要读图处理一遍了
        print(f'# -> Active Cropper Initing Data {loader.part_id}')
        l, u, r, d = loader.box
        possibles = []
        positions = []
        # 松弛距离
        ml, mu, mr, md = self.adaptor.get_margin(loader)
        for _, x in magic_iter(
                target=min(r, r+mr) - max(l, l-mr),
                unit=100,
                step=100,
                margin=(0, 0)
        ):
            for _, y in magic_iter(
                    target=min(d, d+md) - max(u, u-mu),
                    unit=100,
                    step=100,
                    margin=(0, 0)
            ):
                piece = loader.label({
                    'x': l + x,
                    'y': u + y,
                    'degree': 0,
                    'scaling': 1,
                    'w': self.patch_size,
                    'h': self.patch_size,
                })
                sum_p = 0
                for i, p in enumerate(self.weights):
                    sum_p += (piece == i).sum() * p
                    possibles.append(sum_p)
                    positions.append((l + x + 50, u + y + 50))
        # 概率数组加和，得到每个坐标对应的概率区间
        selections = np.cumsum(possibles[:-1])
        # 首项概率 s[1] - s[0]， 次项概率 s[2] - s[1]， 以此类推，最后一项的概率是 s_max - s[-1]
        selections = np.pad(selections, [(1, 0)]).tolist()
        # s_max = s[-1] + p[-1]， 由于 python 的 random 两端包含， 额外 -1， 得到 s_max 的工程值
        selection_max = int(selections[-1] + possibles[-1] - 1)
        result = {
            'selections': selections,
            'positions': positions,
            'max': selection_max,
        }
        # 保存下来
        self.cache[loader.part_id] = result
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(result, f, indent=4)
        return result

    def sample(self, loader: Loader, num: int = -1):
        # 理想区域
        l, u, r, d = self.adaptor.get_box(loader)
        result = []
        # 随机遍历(定数量遍历)
        if num == -1:
            area = (r - l) * (d - u)
            num = math.ceil(area / self.patch_size ** 2)
        active = self.load(loader)
        while len(result) < num:
            # 采样所需的角度、缩放信息
            degrees = self.adaptor.get_rotate_degrees()
            scalings = self.adaptor.get_scaling_rates()
            for degree in degrees:
                # 采样变换后的窗口大小、步长
                rotate_size_rate = Rotator.size_rate(degree)
                for scaling in scalings:
                    # 采样边距
                    padding = math.ceil(self.patch_size * scaling * (rotate_size_rate - 1))
                    # 获得随机概率(整数)
                    active_value = np.random.randint(0, active['max'])
                    # 二分坐标查找
                    active_index = bisect.bisect_right(active['selections'], active_value) - 1
                    # 获得锚点信息
                    x, y = active['positions'][active_index]
                    # 生成随机浮动(高斯分布)
                    fx = np.random.randn() * 25
                    fy = np.random.randn() * 25
                    fx = np.clip(fx, -50, 49)
                    fy = np.clip(fy, -50, 49)
                    result.append({
                        'part_id': loader.part_id,
                        'name': loader.name,
                        'i': -1,
                        'j': -1,
                        # (x, y) 记录中点坐标
                        'x': x + fx,
                        'y': y + fy,
                        'w': self.patch_size,
                        'h': self.patch_size,
                        'degree': degree,
                        'scaling': scaling,
                    })
        return result[:num]
