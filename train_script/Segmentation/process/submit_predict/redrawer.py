import math
from typing import Tuple, Iterable, List, Dict
import cv2
import numpy as np
import torch
from tqdm import tqdm

from utils import Shape
from .asap_slide import Slide as Reader, Writer


def redraw(
    argmaxed: Reader,                   # 输入流
    mask: Reader,                       # 辅助输入流
    contours: List[Shape],              # 轮廓标记
    attentions: List[bool],             # 轮廓归类
    index_matrix: np.ndarray,           # 序数矩阵
    writer: Writer,                     # 输出流
    dimensions: Tuple[int, int],        # 尺寸
    tile_size: int,                     # 写图核尺寸
) -> Tuple[np.ndarray, Shape]:

    W, H = dimensions
    # 图片的宽高对 tile_size 整数倍向上取整
    W = math.ceil(W / tile_size) * tile_size
    H = math.ceil(H / tile_size) * tile_size

    for j, y in enumerate(range(0, H, tile_size)):
        # 获得预测结果
        result = argmaxed.read_region(
            location=(0, y),
            level=0,
            size=(W, tile_size),
        )[:, :, 0]
        for i, x in enumerate(range(0, W, tile_size)):
            patch = result[x:x + tile_size]
            # 依据坐标阵列嗅探轮廓并绘制
            if index_matrix[j, i] is not None:
                for index in index_matrix[j, i]:

                    # c
                    contour = contours[index] - (x, y)
                    # p, [q]
                    outer, inners = contour.sep_p()
                    # [p, *qs]
                    coords = [np.array(pts, dtype=int) for pts in (outer, *inners)]
                    # cv2.fillPoly(patch, coords, draw_value)

                    # TODO: 这里正在魔改
                    if attentions[index]:
                        # 若注意力返回值为 True， 轮廓为浸润性癌，取值 1
                        draw_value = 1
                    else:
                        # 否则依据轮廓内像素数计算结论
                        temp = np.zeros_like(patch)
                        cv2.fillPoly(temp, coords, 1)
                        counter = patch * temp
                        tms = (counter == 3).sum()
                        nms = (counter == 4).sum()
                        # 若统计正常数超过肿瘤数的两倍，视为正常
                        if nms > tms * 2:
                            draw_value = 4
                        else:
                            draw_value = 3
                    cv2.fillPoly(patch, coords, draw_value)
            # 输出
            writer.write(tile=patch, x=x, y=y)


def redraw__for_patch(
    argmaxed: Reader,                   # 输入流
    mask: Reader,                       # 辅助输入流
    contours: List[Shape],              # 轮廓标记
    attentions: List[bool],             # 轮廓归类
    index_matrix: np.ndarray,           # 序数矩阵
    writer: Writer,                     # 输出流
    dimensions: Tuple[int, int],        # 尺寸
    tile_size: int,                     # 写图核尺寸
    label_mapper: Dict[bool, int]       # 轮廓映射表
) -> Tuple[np.ndarray, Shape]:

    W, H = dimensions
    # 图片的宽高对 tile_size 整数倍向上取整
    W = math.ceil(W / tile_size) * tile_size
    H = math.ceil(H / tile_size) * tile_size

    # 逐片迭代
    for i, j, x, y in (
        (_i, _j, _x, _y) for _i, _x in enumerate(range(0, W, tile_size)) for _j, _y in enumerate(range(0, H, tile_size))
    ):
        msk = mask.read_region(
            location=(x, y),
            level=0,
            size=(tile_size, tile_size),
        )
        if not np.any(msk):
            continue

        # 获得预测结果
        result = argmaxed.read_region(
            location=(x, y),
            level=0,
            size=(tile_size, tile_size),
        )

        # 依据坐标阵列嗅探轮廓并绘制
        if index_matrix[j, i] is not None:
            for index in index_matrix[j, i]:
                contour = contours[index] - (x, y)
                outer, inners = contour.sep_p()
                coords = [np.array(pts, dtype=int) for pts in (outer, *inners)]
                draw_value = label_mapper[attentions[index]]
                cv2.fillPoly(result, coords, draw_value)

        # 输出
        writer.write(tile=result, x=x, y=y)
