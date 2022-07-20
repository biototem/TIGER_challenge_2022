from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np

from .rotate import rotate, SlideRotateCropper
from ..model import gaussian_kernel


class Merger(object):
    def __init__(self, shape: Tuple[int, int, int], xyc_dim: Tuple[int, int, int] = (0, 1, 2), kernel_steep: float = 2):
        self.x_dim, self.y_dim, self.c_dim = xyc_dim
        self.W, self.H, self.C = shape[self.x_dim], shape[self.y_dim], shape[self.c_dim]
        self.helper = np.zeros(shape=(self.H, self.W, 1), dtype=np.float32) + 1e-17
        self.target = np.zeros(shape=(self.H, self.W, self.C), dtype=np.float32)
        self.kernel_steep = kernel_steep
        self.kns = {}

    def __kernel__(self, size):
        if size not in self.kns:
            self.kns[size] = np.expand_dims(gaussian_kernel(size=size, steep=self.kernel_steep), 2)
        return self.kns[size]

    def set(self, data, site: Tuple[int, int], size: int, degree: float = 0, scale: float = 1):
        data = data.transpose((self.y_dim, self.x_dim, self.c_dim))
        # 计算图
        kernel = self.__kernel__(size)
        target = data * kernel
        kernel = rotate(kernel, degree, scale)
        target = rotate(target, degree, scale)
        # 计算原图坐标
        h, w = kernel.shape[:2]
        x0, y0 = site
        x1, y1 = round(x0 - w / 2), round(y0 - h / 2)
        # 拼图
        ml = max(0, x1)
        mu = max(0, y1)
        mr = min(self.W, x1+w)
        md = min(self.H, y1+h)
        pl = max(0, -x1)
        pu = max(0, -y1)
        pr = min(w, self.W-x1)
        pd = min(h, self.H-y1)
        if self.target[mu:md, ml:mr, :].shape != target[pu:pd, pl:pr, :].shape:
            print(ml, mu, mr, md, pl, pu, pr, pd)
        self.target[mu:md, ml:mr, :] += target[pu:pd, pl:pr, :]
        self.helper[mu:md, ml:mr, :] += kernel[pu:pd, pl:pr, :]

    def tail(self):
        return self.target / self.helper


def test():
    tif_path = '/media/totem_disk/totem/guozunhu/Project/kidney_biopsy/tiff_data/tif/H1804782 1 HE/H1804782 1 HE_Wholeslide_默认_Extended.tif'
    cp = SlideRotateCropper(filepath=tif_path)
    P = 8000
    S = 1000

    merger = Merger(shape=(S, 3, S), xyc_dim=(0, 2, 1), kernel_steep=8)
    for i in range(55):
        x = np.random.randint(0, S)
        y = np.random.randint(0, S)
        degree = np.random.randint(-44, 45)
        scale = np.random.random() * 3 + 0.25
        img = cp.get(site=(P + x, P + y), size=128, degree=degree, scale=scale)[:, :, :3].transpose((1, 2, 0))
        merger.set(img, site=(x, y), size=128, degree=-degree, scale=1/scale)
        tl = merger.tail().astype(np.uint8)
        plt.imshow(tl)
        plt.show()


if __name__ == '__main__':
    test()
