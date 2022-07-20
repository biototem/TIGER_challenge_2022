import math
import time
from typing import Tuple, Iterable, List
import cv2
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils import Shape, ComplexMultiPolygon, SimplePolygon, SimpleMultiPolygon, Timer
from .asap_slide import Slide as Reader
from .wt import wt_interface_trans_img


def predict(
        model: torch.nn.Module,  # 模型文件
        image: Reader,  # 输入流
        mask: Reader,  # 辅助输入流
        dimensions: Tuple[int, int],  # 维度数
        tile_size: int,  # 写图核尺寸
        tile_step: int,  # 融合步长(tile_size 的整数倍)
        ksize: Tuple[int, int],  # 预测核尺寸
        channel: int,  # 输出通道数
        batch_size: int,  # 预测批数
        device: str,  # 预测所用容器
        T: Timer = None
) -> Tuple[np.ndarray, np.ndarray, Shape, Shape]:
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

    # 下列 H 的计算，为保证 step | (H - kh)
    img_loader = get_loader(H=H + (-H % tile_step), W=W, tile_size=tile_size, reader=image)
    msk_loader = get_loader(H=H + (-H % tile_step), W=W, tile_size=tile_size, reader=mask)

    # 输入缓冲区
    img = torch.zeros(
        kh, W, 3,
        dtype=torch.float32,
        requires_grad=False,
    )
    # 蒙版缓冲区
    msk = torch.zeros(
        kh, W,
        dtype=torch.uint8,
        requires_grad=False,
    )
    # 预测缓冲区
    temp = torch.zeros(
        kh, W, channel,
        dtype=torch.float32,
        requires_grad=False,
    )

    # wt_matrix
    # TODO: 这里必须存成矩阵的原因是，在扫描过程中轮廓的索引不固定，等融合完成后，唯一能用于寻找对应图块的就是矩阵坐标了
    character_matrix = np.empty(shape=(math.ceil(H / tile_size), math.ceil(W / tile_size)), dtype=object)
    # TODO: 于此同时，照顾到特征矩阵的稀疏性，需要用一个坐标列存储全部非 None 的特征坐标
    adjacencies = []

    # contours
    contours = []
    stromas = []

    # 预读行
    for py in range(tile_step, kh, tile_size):
        img[py:py + tile_size, :, :] = next(img_loader).squeeze()
    for py in range(tile_step, kh, tile_size):
        msk[py:py + tile_size, :] = next(msk_loader).squeeze()

    # 进入循环体
    if T: T.track(' -> start predict')
    y: int = 0
    for y in tqdm(range(0, H - kh + tile_step, tile_step), desc='predicting'):
        # if not(40 <= y // kh <= 42):
        #     continue

        t1 = t2 = t3 = 0

        t = time.time()
        # 只保留上下行交叠区域的大小: 恰好等于行高减步长
        img[:kh - tile_step, :, :] = img[tile_step:, :, :]
        # 下行非交叠区域读新数据
        for py in range(tile_step, kh, tile_size):
            img[py:py + tile_size, :, :] = next(img_loader).squeeze()

        # 只保留上下行交叠区域的大小: 恰好等于行高减步长
        msk[:kh - tile_step, :] = msk[tile_step:, :]
        # 下行非交叠区域读新数据
        for py in range(tile_step, kh, tile_size):
            msk[py:py + tile_size, :] = next(msk_loader).squeeze()

        # 只保留上下行交叠区域的大小: 恰好等于行高减步长
        temp[:kh - tile_step, :] = temp[tile_step:, :]
        # 下行非交叠区域直接置 0
        temp[tile_step:, :] = 0
        t1 += time.time() - t

        t = time.time()
        # predict and merge
        pos = [x for x in range(0, W - kw + 1, tile_step) if msk[:, x: x+kw].any()]
        for xs, inputs in get_batch(pos=pos, kw=kw, temp=img, batch_size=batch_size):
            results = model(inputs.to(device))
            # results = torch.empty(batch_size, channel, kh, kw, device=device, requires_grad=False)
            results *= kernel
            results = results.permute(0, 2, 3, 1).detach().squeeze().cpu()
            for x, result in zip(xs, results):
                temp[:, x: x+kw, :] += result
            del results, inputs
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()

        t2 += time.time() - t

        t = time.time()
        argmaxed = temp[:tile_step, :, :].numpy().argmax(axis=2) * msk[:tile_step, :].numpy().astype(bool)
        # argmaxed = temp[:tile_step, :, :].argmax(axis=2) * msk[:tile_step, :].type(torch.bool)
        if argmaxed.any():
            counts = (argmaxed == 1) | (argmaxed == 3) | (argmaxed == 4)
            # TODO: 若 3, 文泰需要包含组织轮廓内的全部图块作为特征进行分析, 使用下列代码
            for py in range(0, tile_step, tile_size):
                for x in range(0, W, tile_size):
                    tile = counts[py:py+tile_size, x: x+tile_size].astype(np.float32)
                    if tile.any() and tile.mean() > 0.05:
                        patch = img[py:py+tile_size, x: x+tile_size, :]
                        j, i = (y + py) // tile_size, x // tile_size
                        character_matrix[j, i] = wt_interface_trans_img(patch)
                        adjacencies.append([i, j])
            # TODO: END ...

            # find_contour
            contour = find_contours(counts) + (0, y)
            contour = contour.buffer(distance=1)
            contour = contour.simplify(tolerance=1)
            contours.extend(contour.sep_out())

            counts = (argmaxed == 2) | (argmaxed == 6)
            stroma = find_contours(counts) + (0, y)
            stroma = stroma.buffer(distance=1)
            stroma = stroma.simplify(tolerance=1)
            stromas.extend(stroma.sep_out())

        t3 += time.time() - t
        print(f'time cost: t1:{t1}, t2:{t2}, t3:{t3}')

    # TODO: 若 3, 文泰需要包含组织轮廓内的全部图块作为特征进行分析, 使用下列代码
    argmaxed = temp[tile_step:, :, :].numpy().argmax(axis=2) * msk[tile_step:, :].numpy().astype(bool)
    # argmaxed = temp[tile_step:, :, :].argmax(axis=2) * msk[tile_step:, :].type(torch.bool)
    if argmaxed.any():
        counts = (argmaxed == 1) | (argmaxed == 3) | (argmaxed == 4)
        for py in range(tile_step, ksize, tile_size):
            for x in range(0, W, tile_size):
                tile = counts[py:py+tile_size, x: x+tile_size].astype(np.float32)
                if tile.any() and tile.mean() > 0.05:
                    patch = img[py:py+tile_size, x: x+tile_size, :]
                    j, i = (y + py) // tile_size, x // tile_size
                    character_matrix[j, i] = wt_interface_trans_img(patch)
                    adjacencies.append([i, j])
        # TODO: END ...

        # find_contour
        contour = find_contours(counts) + (0, y)
        contour = contour.buffer(distance=1)
        contour = contour.simplify(tolerance=1)
        contours.extend(contour.sep_out())

        counts = (argmaxed == 2) | (argmaxed == 6)
        stroma = find_contours(counts) + (0, y)
        stroma = stroma.buffer(distance=1)
        stroma = stroma.simplify(tolerance=1)
        stromas.extend(stroma.sep_out())

    if T: T.track(' -> start contour merge')
    contours = ComplexMultiPolygon(singles=contours)
    contours = contours.standard()
    contours = contours.buffer(distance=-1)
    contours = contours.simplify(tolerance=1)

    stromas = ComplexMultiPolygon(singles=stromas)
    stromas = stromas.standard()
    stromas = stromas.buffer(distance=-1)
    stromas = stromas.simplify(tolerance=1)

    if T: T.track(' -> end predict')
    return character_matrix, adjacencies, contours, stromas


def get_kernel(width: int, height: int, steep: float):
    # create gaussian kernel
    kernel_x = cv2.getGaussianKernel(ksize=width, sigma=width / steep)
    kernel_x /= np.average(kernel_x)
    kernel_y = cv2.getGaussianKernel(ksize=height, sigma=height / steep)
    kernel_y /= np.average(kernel_y)
    kernel = np.matmul(kernel_y, kernel_x.T)
    return torch.tensor(kernel, requires_grad=False)


# 逐行读加速
def get_loader(H: int, W: int, tile_size: int, reader: Reader) -> Iterable[torch.Tensor]:
    L = H // tile_size

    class D(object):
        def __len__(self):
            return L

        def __getitem__(self, item: int):
            img = reader.read_region(
                location=(0, item * tile_size),
                level=0,
                size=(W, tile_size),
            )
            return img

    return iter(DataLoader(
        D(),
        batch_size=1,
        num_workers=1,
    ))


def get_batch(pos: List[int], kw: int, temp: torch.Tensor, batch_size: int) -> torch.Tensor:
    L = len(pos)
    m = torch.tensor([[[0.485, 0.456, 0.406]]])
    s = torch.tensor([[[0.229, 0.224, 0.225]]])

    class D(object):
        def __len__(self):
            return L

        def __getitem__(self, item: int):
            x = pos[item]
            img = temp[:, x: x+kw, :]
            img = (img / 255 - m) / s
            return x, img.permute(2, 0, 1)

    return iter(DataLoader(
        D(),
        batch_size=batch_size,
        num_workers=1,
    ))


def find_contours(temp: np.ndarray) -> Shape:

    # 热图预处理
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
