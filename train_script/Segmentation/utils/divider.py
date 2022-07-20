from typing import Tuple
import numpy as np
from cv2 import distanceTransform as eucl_distance_cv
from skimage.morphology import reconstruction, erosion
from skimage.measure import label
from skimage.segmentation import watershed

from utils import Timer


class Watershed(object):
    def __init__(self):
        self.distance = 0
        self.join = 999

    def with_distance(self, distance: float):
        self.distance = distance
        return self

    def with_join(self, join: float):
        self.join = join
        return self

    def process(self, mask: np.ndarray, returns: Tuple[str, ...] = ('full',), T: Timer = None):
        """
        returns 约定返回值, 默认只返回分水岭结果
        """
        # 蒙版图 -> 二值图
        mask = mask.astype(bool)
        # 距离图 -> 负图
        # TODO: ----效率节点在这个欧几里得距离图计算上----有待优化
        # distmap = - eucl_distance(mask)
        distmap = - eucl_distance_cv(mask.astype(np.uint8), 1, 5)
        if T: T.track(' -> distmap created...')
        # 全局斩峰 -> 避免轮廓完整的大图被分割为不同区域
        distmap[distmap < -self.join] = - self.join
        # 局部斩峰 -> 将同一峡谷内的不同洼地连通起来
        connected = reconstruction(seed=distmap + self.distance, mask=distmap, method='erosion')
        # 局部极值图 -> 获取局部极值点
        local = reconstruction(seed=connected + 1, mask=connected, method='erosion') - connected
        local = local >= 1
        local[~mask] = 0
        if T: T.track(' -> reconstructed...')
        # 种子标记图
        seed = label(local)
        # 分水岭分割图
        watered = watershed(connected, markers=seed, mask=mask, compactness=1, watershed_line=True)
        if T: T.track(' -> watered...')
        # ???
        # watered = arrange_label(what)
        # border
        # border = ~watered.astype(bool) & dilation(watered, selem=np.ones(shape=(3, 3))).astype(bool)
        border = watered.astype(bool) & ~erosion(watered, selem=np.ones(shape=(3, 3))).astype(bool)
        #full
        full = watered.copy()
        full[border] = -1
        if T: T.track(' -> border built...')

        result = {
            'mask': mask,
            'distmap': distmap,
            'connected': connected,
            'local': local,
            'seed': seed,
            'watered': watered,
            'border': border,
            'full': full,
        }
        # titles, imgs = zip(*result.items())
        # PPlot().title(*titles).add(*imgs).show()
        return (result[key] for key in returns)


def arrange_label(mat):
    """
    Given a labelled 2D matrix, returns a labelled 2D matrix
    where it tries to se the background to 0.
    Args:
        mat: 2-D labelled matrix.
    Returns:
        A labelled matrix. Where each connected component is
        assigned an integer and the background is assigned 0.
    """
    val, counts = np.unique(mat, return_counts=True)
    background_val = val[np.argmax(counts)]
    mat = label(mat, background=background_val)
    if np.min(mat) < 0:
        mat += np.min(mat)
        mat = arrange_label(mat)
    return mat


'''
    def process(self, probability: np.ndarray, mask: np.ndarray = None, returns: Tuple[str, ...] = ('watered',)):
        """
        probability 表示概率值图, 它将在 threshold 和 h 的作用下被自动计算为 distmap
        mask 是可选参数,当给定 mask 时, threshold 将失效
        returns 约定返回值, 默认只返回分水岭结果
        """
        # 蒙版图 -> 二值图
        if mask is None:
            mask = probability > self.threshold
        mask = mask.astype(bool)
        # 距离图 -> 负图
        distmap = - eucl_distance(mask)
        # 全局斩峰 -> 避免轮廓完整的大图被分割为不同区域
        distmap[distmap < -self.join] = - self.join
        # 局部斩峰 -> 将同一峡谷内的不同洼地连通起来
        connected = reconstruction(seed=distmap + self.distance, mask=distmap, method='erosion')
        # 局部极值图 -> 获取局部极值点
        local = reconstruction(seed=connected + 1, mask=connected, method='erosion') - connected
        local = local >= 1
        local[~mask] = 0
        # 种子标记图
        seed = label(local)
        # 分水岭分割图
        what = watershed(connected, markers=seed, mask=mask, compactness=1, watershed_line=True)
        # ???
        watered = arrange_label(what)
        # border
        # border = ~watered.astype(bool) & dilation(watered, selem=np.ones(shape=(3, 3))).astype(bool)
        border = watered.astype(bool) & ~erosion(watered, selem=np.ones(shape=(3, 3))).astype(bool)
        #full
        full = watered.copy()
        full[border] = -1

        result = {
            'probability': probability,
            'mask': mask,
            'distmap': distmap,
            'connected': connected,
            'local': local,
            'seed': seed,
            'what': what,
            'watered': watered,
            'border': border,
            'full': full,
        }
        # titles, imgs = zip(*result.items())
        # PPlot().title(*titles).add(*imgs).show()
        return (result[key] for key in returns)
'''
