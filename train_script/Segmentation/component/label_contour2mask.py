from typing import Tuple, List
from scipy import signal
import cv2
import numpy as np

from utils import Shape


LABEL_DICT = {
    'STROMA': 1,
    'NORMAL': 2,
    'TUMOR': 3,
    'NECROSIS': 4,
    'VESSEL': 5,
    'OTHER': 6,
}


def labels2mask(size: Tuple[int, int], labels=List[Tuple[str, Shape]]):
    w, h = size
    temp = np.zeros(shape=(h, w), dtype=np.uint8)
    # 先画大轮廓后画小轮廓，以此支持“轮廓A在轮廓B中”的需求
    labels = sorted(labels, key=lambda pair: pair[1].area, reverse=True)
    for cls, label in labels:
        cls_value = LABEL_DICT[cls]
        for lb in label.sep_out():
            outer, inners = lb.sep_p()
            coords = [np.array(pts, dtype=int) for pts in (outer, *inners)]
            cv2.fillPoly(temp, coords, [cls_value])
            # cv2.drawContours(temp, coords, None, [cls_value])
            # PPlot().add(temp).show()
    return temp


def image2mask(image: np.ndarray):
    # 阈值蒙版
    # # tim1 -> 滤除纯白区域，要求至少在某一RGB上具有小于210的色值
    # tim1 = np.any(image <= 210, 2).astype(np.uint8)
    # # tim2 -> 滤除灰度区域，要求RGB的相对色差大于18
    # tim2 = (np.max(image, 2) - np.min(image, 2) > 18).astype(np.uint8)

    # 先缩放
    h, w, _ = image.shape
    temp = cv2.resize(image, (w // 4, h // 4))
    tim1 = np.any(temp <= 238, 2).astype(np.uint8)
    # tim2 = (np.max(temp, 2) - np.min(temp, 2) > 2).astype(np.uint8)
    m1 = np.max(temp, 2)
    m2 = np.min(temp, 2)
    m3 = np.mean(temp, 2)
    m4 = m2 / 255 * (255-m1)
    m5 = np.clip(m4, 1, 8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m6 = signal.convolve2d(m5, k, mode='same') / k.sum()
    # print(m1.mean(),m2.mean(),m4.mean(),m5.mean())
    tim2 = (m1 - m2 > m6).astype(np.uint8)
    tim = tim1 * tim2
    # 捕获边缘
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # cv2.erode(tim, k, dst=tim, iterations=2)
    # cv2.dilate(tim, k, dst=tim, iterations=4)
    # cv2.erode(tim, k, dst=tim, iterations=2)
    cv2.dilate(tim, k, dst=tim, iterations=2)
    cv2.erode(tim, k, dst=tim, iterations=4)
    cv2.dilate(tim, k, dst=tim, iterations=2)

    tim = cv2.resize(tim, (w, h))
    # from utils import PPlot
    # # PPlot().add(temp, tim1, m1-m2, m5, tim2).show()
    # PPlot().add(temp, tim1, tim2, tim).show()
    return tim

