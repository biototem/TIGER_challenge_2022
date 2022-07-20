from basic import config
from utils import Rotator
from .interface import Cropper
from ..adaptor import Adaptor
from ..loader import Loader


# 不切图，直接缩放
class NoCropper(Cropper):
    def __init__(self, adaptor: Adaptor):
        self.adaptor = adaptor
        self.patch_size = config['cropper.patch_size']

    def sample(self, loader: Loader, num: int):
        # 理想区域
        l, u, r, d = self.adaptor.get_box(loader)
        # 采样所需的角度信息
        degrees = self.adaptor.get_rotate_degrees()
        result = []
        for degree in degrees:
            # 缩放对此无效 -> 一律直接缩放至目标大小
            rotate_size_rate = Rotator.size_rate(degree)
            w, h = loader.size
            scaling = max(
                w * rotate_size_rate / self.patch_size,
                h * rotate_size_rate / self.patch_size,
            )
            result.append({
                'part_id': loader.part_id,
                'name': loader.name,
                'i': 0,
                'j': 0,
                # (x, y) 记录中点坐标
                'x': w//2,
                'y': h//2,
                'w': self.patch_size,
                'h': self.patch_size,
                'degree': degree,
                'scaling': scaling,
            })
        return result
