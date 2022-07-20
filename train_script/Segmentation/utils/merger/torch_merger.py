from typing import Tuple, Any
import cv2
import numpy as np
import torch

from ..model import gaussian_kernel


class TorchMerger(object):
    def __init__(self, class_num: int = 8, kernel_size: int = 256, kernel_steep: float = 2, zoom: float = 1):
        # self.kns = {self.__kernel__(zoom, steep) for zoom, steep in kernel_params.items()}
        self.C = class_num
        self.ksize = kernel_size
        self.W = self.H = 0
        self.target = self.helper = None
        self.zoom = zoom
        # kernel 用于 同预测结果相乘
        self.kernel = self.__kernel__(steep=kernel_steep)
        # kns 用于 对齐切图方法
        self.__kns__ = {}

    def __kernel__(self, steep):
        kernel = gaussian_kernel(size=self.ksize, steep=steep)
        kernel = np.expand_dims(kernel, axis=(0, 1))
        return torch.tensor(kernel).cuda()

    def with_shape(self, W: int, H: int):
        self.W = W
        self.H = H
        return self

    def set(self, patches, grids):
        # 计算图
        targets = patches * self.kernel
        targets = targets.cpu().numpy().transpose((0, 2, 3, 1))
        # x,y 标定原图起止点坐标,size标定原图切图尺寸(用于缩放),当grids中不存在size属性时,默认全部按照patch_size切图(对应缩放为1)
        for x, y, target in zip(grids['x'], grids['y'], targets):
            x, y = int(x / self.zoom), int(y / self.zoom)
            size = self.ksize // self.zoom
            # (x, y) 为图片中心，而此处需要将其处理为左上角坐标
            x -= size // 2
            y -= size // 2
            if size != self.ksize:
                target = cv2.resize(target, (size, size))
            kernel = self.kns(size)
            temp_left = max(0, x)
            temp_up = max(0, y)
            temp_right = min(self.W, x + size)
            temp_down = min(self.H, y + size)
            patch_left = max(0, -x)
            patch_up = max(0, -y)
            patch_right = min(size, self.W - x)
            patch_down = min(size, self.H - y)
            self.target[temp_up: temp_down, temp_left: temp_right, :] += target[patch_up: patch_down, patch_left: patch_right, :]
            self.helper[temp_up: temp_down, temp_left: temp_right, :] += kernel[patch_up: patch_down, patch_left: patch_right, :]

    def tail(self):
        return self.target / self.helper

    def kns(self, size: int):
        if size not in self.__kns__:
            kn = self.kernel[0, 0, :, :].cpu().numpy()
            self.__kns__[size] = np.expand_dims(cv2.resize(kn, (size, size)), axis=2)
        return self.__kns__[size]

    def __enter__(self):
        self.helper = np.zeros(shape=(self.H, self.W, 1), dtype=np.float32) + 1e-17
        self.target = np.zeros(shape=(self.H, self.W, self.C), dtype=np.float32)
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any):
        self.W = 0
        self.H = 0
        self.helper = None
        self.target = None
        return False
