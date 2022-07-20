import math
from typing import Tuple, Iterable
import cv2
import numpy as np
import torch
from tqdm import tqdm

from utils import Shape, ComplexMultiPolygon, SimplePolygon, SimpleMultiPolygon, Timer
from .asap_slide import Slide as Reader, Writer
from .wt_interface import wt_interface_trans_img


def predict(
    model: torch.nn.Module,             # 模型文件
    image: Reader,                      # 输入流
    mask: Reader,                       # 辅助输入流
    writer: Writer,                     # 输出流
    dimensions: Tuple[int, int],        # 维度数
    tile_size: int,                     # 写图核尺寸
    tile_step: int,                     # 融合步长(tile_size 的整数倍)
    ksize: Tuple[int, int],             # 预测核尺寸
    channel: int,                       # 输出通道数
    device: str,                        # 预测所用容器
    T: Timer
) -> Tuple[np.ndarray, Shape]:

    W, H = dimensions
    # 图片的宽高对 tile_size 整数倍向上取整
    W = math.ceil(W / tile_size) * tile_size
    H = math.ceil(H / tile_size) * tile_size

    kw, kh = ksize if isinstance(ksize, Iterable) else (ksize, ksize)
    assert min(kw, kh) >= tile_step >= tile_size, '核长不得小于步长, 步长不得小于片长'
    assert tile_step % tile_size == 0, '步长必须是片长的整数倍'
    assert kw % tile_size == 0 and kh % tile_size == 0, '核长必须是片长的整数倍'
    assert tile_size % 16 == 0, '片长必须是 16 的整数倍'
    kernel = get_kernel(width=kw, height=kh, steep=4).to(device).unsqueeze(0).unsqueeze(0)

    model.to(device)

    # wt_matrix
    character_matrix = np.empty(shape=(math.ceil(H / tile_size), math.ceil(W / tile_size)), dtype=object)

    # contours
    contours = []

    # TODO: (默认) 若 1, 文泰需要包含白背景在内的全部图块作为特征进行分析, 使用下列代码
    # for x in range(0, W, tile_size):
    #     for y in range(0, H, tile_size):
    #         img = image.read_region(
    #             location=(x, y),
    #             level=0,
    #             size=(tile_size, tile_size),
    #         )
    #         character_matrix[y // tile_size, x // tile_size] = wt_interface_trans_img(img)
    # TODO: END ...

    # TODO: 若 2, 文泰需要包含组织轮廓内的全部图块作为特征进行分析, 使用下列代码
    # for x in range(0, H, tile_size):
    #     for y in range(0, W, tile_size):
    #         msk = mask.read_region(
    #             location=(x, y),
    #             level=0,
    #             size=(kw, kh),
    #         )
    #         if not np.any(msk):
    #             continue
    #         img = image.read_region(
    #             location=(x, y),
    #             level=0,
    #             size=(tile_size, tile_size),
    #         )
    #         character_matrix[y // tile_size, x // tile_size] = wt_interface_trans_img(img)
    # TODO: END ...

    # 按 行 / y 缓存, 行高 == 核高, 行长 == 图宽
    temp = np.zeros(
        shape=(kh, W, channel),
        dtype=np.float16
    )

    if T: T.track(' -> start predict')
    # loop over image and get tiles
    for y in tqdm(range(0, H, tile_step), desc='predicting'):
        # 只保留上下行交叠区域的大小: 恰好等于行高减步长
        temp[:kh-tile_step, :] = temp[tile_step:, :]
        # 下行非交叠区域直接置 0
        temp[tile_step:, :] = 0
        # predict and merge
        for x in range(0, W - kw + 1, tile_step):
            # mpp = 0.5
            # mpp = 0.25, level = 0, x,y,kw,kh
            msk = mask.read_region(
                location=(x, y),
                level=0,
                size=(kw, kh),
            )
            if not np.any(msk):
                continue

            img = image.read_region(
                location=(x, y),
                level=0,
                size=(kw, kh),
            )

            img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2) / 255

            result = model(img.to(device))

            result *= kernel

            result = result.permute(0, 2, 3, 1).detach().squeeze().cpu().numpy()

            temp[:, x: x+ksize, :] += result * msk.astype(bool)

        # write loop
        for py in range(0, tile_step, tile_size):
            for x in range(0, W, tile_size):
                tile = temp[py:py+tile_size, x: x+tile_size].argmax(axis=2)
                # TODO: 若 3, 文泰需要包含组织轮廓内的全部图块作为特征进行分析, 使用下列代码
                # count = (tile == 1).sum() + (tile == 3).sum() + (tile == 4).sum()
                # if count / tile.size > 0.05:
                #     img = image.read_region(
                #         location=(x, y + py),
                #         level=0,
                #         size=(tile_size, tile_size),
                #     )
                #     character_matrix[(y + py) // tile_size, x // tile_size] = wt_interface_trans_img(img)
                # TODO: END ...
                writer.write(tile=tile, x=x, y=y + py)
        # find_contour
        # mc
        contour = find_contours(temp[:tile_step, :].argmax(axis=2)) + (0, y)
        contour = contour.buffer(distance=1)
        contour = contour.simplify(tolerance=1)
        # mc -> [c]
        contours.extend(contour.sep_out())
        # contours = contours | (find_contour(temp[:tile_step, :].argmax(axis=2)) + (0, y))

    # write end
    for py in range(tile_step, ksize, tile_size):
        for x in range(0, W, tile_size):
            tile = temp[py:py+tile_size, x: x+tile_size].argmax(axis=2)
            # TODO: 若 3, 文泰需要包含组织轮廓内的全部图块作为特征进行分析, 使用下列代码
            # count = (tile == 1).sum() + (tile == 3).sum() + (tile == 4).sum()
            # if count / tile.size > 0.05:
            #     img = image.read_region(
            #         location=(x, y + py),
            #         level=0,
            #         size=(tile_size, tile_size),
            #     )
            #     character_matrix[(y + py) // tile_size, x // tile_size] = wt_interface_trans_img(img)
            # TODO: END ...
            writer.write(tile=tile, x=x, y=y + py)
    # find_contour
    contour = find_contours(temp[tile_step:, :].argmax(axis=2)) + (0, y)
    contour = contour.buffer(distance=1)
    contour = contour.simplify(tolerance=1)
    contours.extend(contour.sep_out())
    # contours = contours | (find_contour(temp[:tile_step, :].argmax(axis=2)) + (0, y))

    if T: T.track(' -> start contour merge')
    # [c] -> mc
    contours = ComplexMultiPolygon(singles=contours)
    # 轮廓间相交,不合法 -> 融合后的,合法
    contours = contours.standard()
    contours = contours.buffer(distance=-1)
    contours = contours.simplify(tolerance=1)

    if T: T.track(' -> end predict')
    # mc
    return character_matrix, contours


def get_kernel(width: int, height: int, steep: float):
    # create gaussian kernel
    kernel_x = cv2.getGaussianKernel(ksize=width, sigma=width / steep)
    kernel_x /= np.average(kernel_x)
    kernel_y = cv2.getGaussianKernel(ksize=height, sigma=height / steep)
    kernel_y /= np.average(kernel_y)
    kernel = np.matmul(kernel_y, kernel_x.T)
    return torch.tensor(kernel, requires_grad=False)


def find_contours__tmp(
    argmaxed: Reader,
    mask: Reader,
    tile_size: int,
    dimensions: Tuple[int, int],
) -> Shape:

    W, H = dimensions
    # 图片的宽高对 tile_size 整数倍向上取整
    W = math.ceil(W / tile_size) * tile_size
    H = math.ceil(H / tile_size) * tile_size

    # contours = ComplexMultiPolygon.EMPTY
    contours = []

    # 逐片迭代
    for j, y in [
        (_j, _y) for _j, _y in enumerate(range(0, H, tile_size))
    ]:
        # 获得预测结果
        temp = argmaxed.read_region(
            location=(0, y),
            level=0,
            size=(W, tile_size),
        )

        contour = find_contours(temp[:, :, 0]) + (0, y)
        contour = contour.smooth(distance=5)
        contour = contour.simplify(tolerance=5)
        contours.extend(contour.sep_out())
        # contours = contours | (find_contour(temp[:, :, 0]) + (0, y))
        # contours = contours.smooth(distance=5)
        # contours = contours.simplify(tolerance=5)

    contours = ComplexMultiPolygon(singles=contours)
    contours = contours.standard()
    contours = contours.smooth(distance=15)
    contours = contours.simplify(tolerance=5)

    return contours


def find_contours(temp: np.ndarray) -> Shape:

    # 热图预处理
    temp = (temp == 1) | (temp == 3) | (temp == 4)
    temp = temp.astype(np.uint8)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cv2.dilate(src=temp, kernel=k, dst=temp, iterations=2)
    cv2.erode(src=temp, kernel=k, dst=temp, iterations=4)
    cv2.dilate(src=temp, kernel=k, dst=temp, iterations=2)

    # 轮廓提取
    contours, hierarchy = cv2.findContours(temp, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    # 某些情况下，行中没有任何轮廓
    if hierarchy is None:
        return ComplexMultiPolygon.EMPTY

    # Remove the middle dim
    contours = [contour[:, 0, :] for contour in contours]
    # format the cv2-coords to correct-coords
    # 这是一个可怕的困难任务, 我已经在这里浪费了一天多的时间, 看起来经过一大堆缜密的"调参", 它现在终于稳定了.
    # 警告: 下面这行代码强依赖于 cv2.CHAIN_APPROX_NONE
    contours = [contour[sorted(np.unique(contour, axis=0, return_index=True)[1])] for contour in contours]
    # rs = []
    # for contour in contours:
    #     _, indexes = np.unique(contour, axis=0, return_index=True)
    #     indexes.sort()
    #     rs.append(contour[indexes])
    # contours = rs

    contours = [contour.tolist() for contour in contours]
    hierarchy = hierarchy[0, :, :].tolist()

    for coords in contours:
        assert SimplePolygon(outer=coords).is_valid()
        # if not SimplePolygon(outer=coords).is_valid():
        #     from utils import PPlot
        #     PPlot().add(SimplePolygon(outer=coords)).show()

    indexes = [-1] * len(contours)
    outers = []
    inners = []

    for index, (contour, hie) in enumerate(zip(contours, hierarchy)):
        if hie[-1] == -1:
            indexes[index] = len(outers)
            outers.append(contour)
    for index, (contour, hie) in enumerate(zip(contours, hierarchy)):
        if hie[-1] != -1:
            inners.append(contour)

    # Create inners and outers
    outer = SimpleMultiPolygon(outers=outers).standard()
    inner = SimpleMultiPolygon(outers=inners).standard()
    shape = outer >> inner
    # shape = shape.smooth(distance=5)
    # shape = shape.simplify(tolerance=5)
    assert shape.is_valid(), '?'
    return shape
