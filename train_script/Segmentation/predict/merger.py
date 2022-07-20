from typing import Tuple, Any
import cv2
import numpy as np
import torch

from basic import hyp
from utils import gaussian_kernel


class TorchMerger(object):
    def __init__(self, channel: int = 0, ksize: int = 0, kernel_steep: float = 2, scales: Tuple[float, ...] = (1.0,)):
        # self.kns = {self.__kernel__(zoom, steep) for zoom, steep in kernel_params.items()}
        self.C = channel or hyp['dataset.class_num']
        self.ksize = ksize or hyp['cropper.patch_size']
        self.W = self.H = 0
        self.target = self.helper = None
        # kernel 用于 同预测结果相乘
        self.kernel = self.__kernel__(steep=kernel_steep)
        # kns 用于 对齐切图方法
        self.__kns__ = {}
        # kn = self.kernel[0, 0, :, :].cpu().numpy()
        # for scale in scales:
        #     scale = np.round(scale, decimals=2)
        #     u = round(self.ksize * scale)
        #     self.kns[u] = np.expand_dims(cv2.resize(kn, (u, u)), axis=2)

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
        if 'size' not in grids:
            grids['size'] = [self.ksize] * len(targets)
        for x, y, size, target in zip(grids['x'], grids['y'], grids['size'], targets):
            x, y, size = int(x), int(y), int(size)
            if size != self.ksize:
                target = cv2.resize(target, (size, size))
            kernel = self.kns(size)
            self.target[y:y + size, x:x + size, :] += target
            self.helper[y:y + size, x:x + size, :] += kernel

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
