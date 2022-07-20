import math
from typing import Tuple, List
import cv2
import numpy as np

from utils import Shape, Region
from .asap_slide import Writer


def redraw(
    contours: List[Shape],              # 轮廓标记
    attentions: List[bool],             # 轮廓归类
    stromas: List[Shape],               # 间质轮廓
    writer: Writer,                     # 输出流
    dimensions: Tuple[int, int],        # 尺寸
    tile_size: int,                     # 写图核尺寸
) -> Tuple[np.ndarray, Shape]:

    W, H = dimensions
    # 图片的宽高对 tile_size 整数倍向上取整
    W = math.ceil(W / tile_size) * tile_size
    H = math.ceil(H / tile_size) * tile_size

    # 先生成轮廓访问阵列
    index_matrix_contour,  index_matrix_stroma = build_map(dimensions=dimensions, contours=contours, stromas=stromas, tile_size=tile_size)

    for j, y in enumerate(range(0, H, tile_size)):
        for i, x in enumerate(range(0, W, tile_size)):
            if index_matrix_contour[j, i] is None and index_matrix_stroma[j, i] is None:
                continue

            tile = np.zeros(shape=(tile_size, tile_size), dtype=np.uint8)
            region = Region(left=i, up=j, right=i+1, down=j+1) * tile_size

            if index_matrix_contour[j, i] is not None:
                for index in index_matrix_contour[j, i]:
                    contour = (contours[index] & region) - (x, y)
                    for c in contour:
                        outer, inners = c.sep_p()
                        coords = [np.array(pts, dtype=int) for pts in (outer, *inners)]
                        # TODO: 这里正在魔改
                        # 若注意力返回值为 True， 轮廓为浸润性癌，取值 1， 否则 取值 3
                        draw_value = 1 if attentions[index] else 3
                        cv2.fillPoly(tile, coords, draw_value)

            if index_matrix_stroma[j, i] is not None:
                for index in index_matrix_stroma[j, i]:
                    # 交集运算会导致 single 变 multi，但是会降低复杂大轮廓平移、绘图的运算量（但不知道交集运算和这俩谁代价更高）
                    contour = (stromas[index] & region) - (x, y)
                    for c in contour:
                        outer, inners = c.sep_p()
                        coords = [np.array(pts, dtype=int) for pts in (outer, *inners)]
                        # 间质返回值 2
                        cv2.fillPoly(tile, coords, 2)
            # 输出
            writer.write(tile=tile, x=x, y=y)


def build_map(dimensions: Tuple[int, int], contours: List[Shape], stromas: List[Shape], tile_size: int):
    W, H = dimensions
    w, h = math.ceil(W / tile_size), math.ceil(H / tile_size)

    contour_matrix = np.empty(shape=(h, w), dtype=object)
    for index, contour in enumerate(contours):
        l, u, r, d = contour.bounds
        l = math.floor(l / tile_size)
        u = math.floor(u / tile_size)
        r = math.ceil(r / tile_size)
        d = math.ceil(d / tile_size)
        for i, j in [(_i, _j) for _i in range(l, r) for _j in range(u, d)]:
            region = Region(left=i, up=j, right=i+1, down=j+1) * tile_size
            if region.is_joint(contour):
                if contour_matrix[j, i] is None:
                    contour_matrix[j, i] = []
                contour_matrix[j, i].append(index)

    stroma_matrix = np.empty(shape=(h, w), dtype=object)
    for index, stroma in enumerate(stromas):
        l, u, r, d = stroma.bounds
        l = math.floor(l / tile_size)
        u = math.floor(u / tile_size)
        r = math.ceil(r / tile_size)
        d = math.ceil(d / tile_size)
        for i, j in [(_i, _j) for _i in range(l, r) for _j in range(u, d)]:
            region = Region(left=i, up=j, right=i + 1, down=j + 1) * tile_size
            if region.is_joint(stroma):
                if stroma_matrix[j, i] is None:
                    stroma_matrix[j, i] = []
                stroma_matrix[j, i].append(index)
    return contour_matrix, stroma_matrix
