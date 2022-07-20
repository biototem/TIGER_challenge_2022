import numpy as np

from utils import Rotator


class SlideLoader(object):
    def __init__(self, path: str, level: int , zoom: float, **kwargs):
        self.loader = Rotator.SlideCropper(filepath=path, level=level)
        self.zoom = zoom

    def __call__(self, grid: dict):
        # scaling -> 数据集缩放
        # zoom -> 数据源缩放
        # 实际加载原图时，两个缩放需要倍乘在一起
        image = self.loader.get(
            site=(grid['x'] * self.zoom, grid['y'] * self.zoom),
            size=(grid['w'], grid['h']),
            degree=grid['degree'],
            scale=grid['scaling'] * self.zoom,
        )
        return np.array(image)[:, :, :3]
