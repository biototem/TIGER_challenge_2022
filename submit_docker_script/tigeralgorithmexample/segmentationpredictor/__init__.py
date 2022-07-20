import math
from typing import Tuple, Iterable, List
import cv2
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import multiresolutionimageinterface as mir

from rw import SegmentationWriter


def seg_predict(
        model: torch.nn.Module,  # 模型文件
        image: mir.MultiResolutionImage,  # 输入流
        mask: mir.MultiResolutionImage,  # 辅助输入流
        writer: SegmentationWriter,  # 输出流
        dimensions: Tuple[int, int],  # 维度数
        tile_size: int,  # 写图核尺寸
        tile_step: int,  # 融合步长(tile_size 的整数倍)
        ksize: Tuple[int, int],  # 预测核尺寸
        channel: int,  # 输出通道数
        batch_size: int,  # 预测批数
        device: str,  # 预测所用容器
        downsample: int,  # 降采样留存倍率
) -> np.ndarray:
    W, H = dimensions
    # 图片的宽高对 tile_size 整数倍向上取整
    W = math.ceil(W / tile_size) * tile_size
    H = math.ceil(H / tile_size) * tile_size
    # W = math.floor(W / tile_size) * tile_size
    # H = math.floor(H / tile_size) * tile_size

    kw, kh = ksize if isinstance(ksize, Iterable) else (ksize, ksize)
    assert min(kw, kh) >= tile_step >= tile_size >= downsample, '核长不得小于步长, 步长不得小于片长，片长不得小于降采样倍率'
    assert tile_step % tile_size == 0, '步长必须是片长的整数倍'
    assert kw % tile_size == 0 and kh % tile_size == 0, '核长必须是片长的整数倍'
    assert tile_size % 16 == 0, '片长必须是 16 的整数倍'
    assert tile_size % downsample == 0, '片长必须是降采样倍率的整数倍'
    kernel = get_kernel(width=kw, height=kh, steep=4).to(device).unsqueeze(0).unsqueeze(0)

    model.to(device)

    # 下列 H 的计算，为保证 step | (H - kh)
    img_loader = get_loader(H=H + (-H % tile_step), W=W, tile_size=tile_size, reader=image, formatter=True)
    msk_loader = get_loader(H=H + (-H % tile_step), W=W, tile_size=tile_size, reader=mask, formatter=False)

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

    # 降采样缩略图
    # sample_matrix = np.zeros(shape=(
    #     H // downsample, W // downsample
    # ), dtype=np.uint8)

    # 预读行
    for py in range(tile_step, kh, tile_size):
        img[py:py + tile_size, :, :] = next(img_loader).squeeze()
    for py in range(tile_step, kh, tile_size):
        msk[py:py + tile_size, :] = next(msk_loader).squeeze()

    # 进入循环体
    y: int = 0
    for y in tqdm(range(0, H - kh + tile_step, tile_step), desc='segmentation predicting'):
        # t1 = t2 = t3 = 0

        # t = time.time()
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
        # t1 += time.time() - t

        # t = time.time()
        # predict and merge
        pos = [x for x in range(0, W - kw + 1, tile_step) if msk[:, x: x+kw].any()]
        for xs, inputs in get_batch(pos=pos, kw=kw, temp=img, batch_size=batch_size):
            try:
                results = model(inputs.to(device))
            except Exception:
                results = torch.empty(batch_size, channel, kh, kw, device=device, requires_grad=False)
            results *= kernel
            results = results.permute(0, 2, 3, 1).detach().squeeze().cpu()

            for x, result in zip(xs, results):
                temp[:, x: x+kw, :] += result
            del results, inputs
            torch.cuda.empty_cache()
            # torch.cuda.empty_cache()
            # torch.cuda.empty_cache()
        # temp *= msk.type(torch.bool).unsqueeze(2)
        # t2 += time.time() - t

        # t = time.time()
        # write loop
        for py in range(0, tile_step, tile_size):
            for x in range(0, W, tile_size):
                m = msk[py:py + tile_size, x: x + tile_size]
                if m.any():
                    tile = temp[py:py + tile_size, x: x + tile_size, :].argmax(dim=2) * m.type(torch.bool)
                    writer.write_segmentation(tile.numpy(), x=x, y=y + py)
                    # add_sample(matrix=sample_matrix, tile=tile, tile_size=tile_size, downsample=downsample, pos=(x, y + py))
                    # writer.write(tile=tile.numpy().astype(np.uint8), x=x, y=y + py)
        # t3 += time.time() - t
        # print(f'time cost: t1:{t1}, t2:{t2}, t3:{t3}')

    # write end
    for py in range(tile_step, ksize, tile_size):
        if y+py >= H: break
        for x in range(0, W, tile_size):
            m = msk[py:py + tile_size, x: x + tile_size]
            if m.any():
                tile = temp[py:py + tile_size, x: x + tile_size, :].argmax(dim=2) * m.type(torch.bool)
                writer.write_segmentation(tile.numpy(), x=x, y=y + py)
                # add_sample(matrix=sample_matrix, tile=tile, tile_size=tile_size, downsample=downsample, pos=(x, y + py))
                # writer.write(tile=tile.numpy().astype(np.uint8), x=x, y=y + py)
    # return sample_matrix


def get_kernel(width: int, height: int, steep: float):
    # create gaussian kernel
    kernel_x = cv2.getGaussianKernel(ksize=width, sigma=width / steep)
    kernel_x /= np.average(kernel_x)
    kernel_y = cv2.getGaussianKernel(ksize=height, sigma=height / steep)
    kernel_y /= np.average(kernel_y)
    kernel = np.matmul(kernel_y, kernel_x.T)
    return torch.tensor(kernel, requires_grad=False)


# 逐行读加速
def get_loader(H: int, W: int, tile_size: int, reader, formatter: bool) -> Iterable[torch.Tensor]:
    L = H // tile_size

    class D(object):
        def __len__(self):
            return L

        def __getitem__(self, item: int):
            img = reader.getUCharPatch(
                startX=0, startY=item * tile_size, width=W, height=tile_size, level=0
            )
            # img = reader.read_region(
            #     location=(0, item * tile_size),
            #     level=0,
            #     size=(W, tile_size),
            # )
            if formatter:
                img = (img / 255 - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
            return img

    return iter(DataLoader(
        D(),
        batch_size=1,
        num_workers=0,
    ))


def get_batch(pos: List[int], kw: int, temp: torch.Tensor, batch_size: int) -> torch.Tensor:
    L = len(pos)

    class D(object):
        def __len__(self):
            return L

        def __getitem__(self, item: int):
            x = pos[item]
            return x, temp[:, x: x+kw, :].permute(2, 0, 1)

    return iter(DataLoader(
        D(),
        batch_size=batch_size,
        num_workers=0,
    ))


def add_sample(matrix: np.ndarray, tile: np.ndarray, tile_size: int, downsample: int, pos: Tuple[int, int]) -> None:
    x, y = pos
    # TODO: 生成波哥需要的采样矩阵
    for qx in range(0, tile_size, downsample):
        for qy in range(0, tile_size, downsample):
            i = (x + qx) // downsample
            j = (y + qy) // downsample
            patch = tile[qy: qy + downsample, qx: qx + downsample]
            # TODO： 采样逻辑，只要有浸润癌，直接浸润； 只要有间质，直接间质； 否则直接 0
            matrix[j, i] = 1 if (patch == 1).any() else 2 if (patch == 2).any() or (patch == 6).any() else 0
