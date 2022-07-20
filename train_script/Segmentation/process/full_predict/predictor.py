import math
from typing import Tuple, Iterable
import cv2
import numpy as np
import torch
from tqdm import tqdm

from .asap_slide import Slide as Reader, Writer


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
) -> None:

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

    # 按 行 / y 缓存, 行高 == 核高, 行长 == 图宽
    temp = np.zeros(
        shape=(kh, W, channel),
        dtype=np.float16
    )

    # loop over image and get tiles
    y: int = 0
    for y in tqdm(range(0, H, tile_step), desc='predicting'):
        # 只保留上下行交叠区域的大小: 恰好等于行高减步长
        temp[:kh-tile_step, :] = temp[tile_step:, :]
        # 下行非交叠区域直接置 0
        temp[tile_step:, :] = 0
        # predict and merge
        for x in range(0, W - kw + 1, tile_step):
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
                writer.write(tile=tile, x=x, y=y + py)

    # write end
    for py in range(tile_step, ksize, tile_size):
        for x in range(0, W, tile_size):
            tile = temp[py:py+tile_size, x: x+tile_size].argmax(axis=2)
            writer.write(tile=tile, x=x, y=y + py)


def get_kernel(width: int, height: int, steep: float):
    # create gaussian kernel
    kernel_x = cv2.getGaussianKernel(ksize=width, sigma=width / steep)
    kernel_x /= np.average(kernel_x)
    kernel_y = cv2.getGaussianKernel(ksize=height, sigma=height / steep)
    kernel_y /= np.average(kernel_y)
    kernel = np.matmul(kernel_y, kernel_x.T)
    return torch.tensor(kernel, requires_grad=False)
