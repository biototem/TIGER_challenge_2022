import numpy as np

from basic import InterfaceProxy
from utils import Watershed, Timer, PPlot


# 这是 zoom = 2 分辨率下的分水岭有效参数
# water = Watershed().with_distance(5).with_join(25)
# 暂时尝试直接放大两倍看看能不能得到 zoom = 1 分辨率下的参数
water = Watershed().with_distance(10).with_join(50)


def divide(predict: np.ndarray, show: bool = False, T: Timer = None):
    p = predict[:, :, 2] + predict[:, :, 3]
    b = predict[:, :, 1]

    m1 = p > 0.3
    m2 = b > 0.2
    m3 = m1 * ~m2
    if T: T.track(f' -> mask created...')

    mask, distmap, seed, full = water.process(mask=m3, T=T, returns=('mask', 'distmap', 'seed', 'full'))
    if show:
        PPlot().title(
            'p>0.3', 'b>0.2', 'mask', 'distmap', 'seed', 'result'
        ).add(
            m1, m2, mask, distmap, seed, full
        ).show()
    return full


def old_divide(predict: np.ndarray, show: bool = False, T: Timer = None):
    p = predict[:, :, 2] + predict[:, :, 3]
    b = predict[:, :, 1]

    m1 = p > 0.3
    m2 = b > 0.2
    m3 = m1 * ~m2

    if T: T.track(' -> water-process')
    dist, seed, full = water.process(mask=m3, returns=('dist', 'seed', 'full'))

    if show:
        if T: T.track(' -> show')
        PPlot().title(
            'p', 'm1=p>0.3', 'm2=b>0.2', 'm3=m1-m2', 'dist=m3.distmap()', 'seed', 'full'
        ).add(
            p, m1, m2, m3, dist, seed, full
        ).show()
    return full
