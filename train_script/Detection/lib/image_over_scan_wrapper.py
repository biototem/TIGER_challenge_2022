'''
图像过采样包装类
'''

import numpy as np
import cv2
from typing import Iterable, Union
try:
    from im_tool import check_and_tr_umat, ensure_image_has_3dim, copy_make_border
except ModuleNotFoundError:
    from .im_tool import check_and_tr_umat, ensure_image_has_3dim, copy_make_border


class ImageOverScanWrapper:
    '''
    图像溢出范围采样类，用于使坐标超出图像边缘时仍然能正常工作
    '''
    def __init__(self, im: np.ndarray):
        assert im.ndim == 3, 'Only support ndim=3 picture, if gray image please add a dim as last.'
        self.im = im

    def get(self, yx_start, yx_end, pad_value: Union[float, int, Iterable]=0):
        im = self.im
        assert len(yx_start) == len(yx_end) == 2, 'Error. Wrong parameters yx_start or yx_parameters'
        assert yx_end[0] > yx_start[0] and yx_end[1] > yx_start[1], 'Error. Not allow get image with size is 0'

        # 这里确保 pad_value 能填满整个通道
        if isinstance(pad_value, Iterable):
            assert len(pad_value) == im.shape[-1], 'Error. Found pad_value is Iterable but asssert len(pad_value) == im.shape[-1] false'
        else:
            pad_value = [pad_value] * im.shape[-1]
        pad_value = tuple(pad_value)

        # 用于处理图像边界问题
        real_yx_start = np.clip(yx_start, [0, 0], im.shape[:2])
        real_yx_end = np.clip(yx_end, [0, 0], im.shape[:2])

        # 额外判断，如果区域完全没有覆盖原图，后面会出错，预先判断后直接填充一个新区域
        if (yx_end[0] <= 0 and yx_end[1] <= 0) or (yx_start[0] >= self.im.shape[0] and yx_start[1] >= self.im.shape[1]) or\
                real_yx_start[0] == real_yx_end[0] or real_yx_start[1] == real_yx_end[1]:
            empty_im = np.empty([yx_end[0]-yx_start[0], yx_end[1]-yx_start[1], im.shape[-1]], self.im.dtype)
            empty_im[:, :, :] = pad_value
            return empty_im

        im2 = im[real_yx_start[0]: real_yx_end[0], real_yx_start[1]: real_yx_end[1]]

        top = max(-yx_start[0], 0)
        left = max(-yx_start[1], 0)
        bottom = max(yx_end[0]-im.shape[0], 0)
        right = max(yx_end[1]-im.shape[1], 0)
        if 0 == top == left == bottom == right:
            return im2
        im3 = check_and_tr_umat(copy_make_border(im2, top, bottom, left, right, value=pad_value))
        assert im3.shape[0] == (yx_end[0] - yx_start[0]) and im3.shape[1] == (yx_end[1] - yx_start[1])
        im3 = ensure_image_has_3dim(im3)
        return im3

    def set(self, yx_start, yx_end, new_im):
        im = self.im
        assert len(yx_start) == len(yx_end) == 2
        assert yx_end[0] > yx_start[0] and yx_end[1] > yx_start[1]
        assert new_im.shape[0] == yx_end[0] - yx_start[0] and new_im.shape[1] == yx_end[1] - yx_start[1]
        assert im.shape[2] == new_im.shape[2]

        # 用于处理图像边界问题
        pr = im
        real_yx_start = np.clip(yx_start, [0, 0], None)
        real_yx_end = np.clip(yx_end, None, pr.shape[:2])

        pr: np.ndarray = pr[real_yx_start[0]: real_yx_end[0], real_yx_start[1]: real_yx_end[1]]

        top, left = real_yx_start - yx_start
        bottom, right = yx_end - real_yx_end

        crop_r = new_im[top: new_im.shape[0] - bottom, left: new_im.shape[1] - right]

        pr[:] = crop_r

    @property
    def data(self):
        return self.im

    @property
    def shape(self):
        return self.im.shape

    @property
    def ndim(self):
        return self.im.ndim

    @property
    def size(self):
        return self.im.size
    
    @property
    def dtype(self):
        return self.im.dtype

    @property
    def itemsize(self):
        return self.im.itemsize
