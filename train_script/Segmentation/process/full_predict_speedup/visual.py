import math
from typing import Tuple, Iterable
import cv2
import numpy as np
import torch
from tqdm import tqdm

from basic import config
from .asap_slide import Slide as Reader


def predict2visual(result: Reader, output_path: str, tile_size: int, downsample: int):

    W, H = result.getDimensions()
    W = math.ceil(W / tile_size) * tile_size
    H = math.ceil(H / tile_size) * tile_size
    w, h = W // downsample, H // downsample

    # 全图可视化的工作
    colors = np.array(config['visual.color.fill'], dtype=np.uint8)
    visual = np.zeros(shape=(h, w, 3), dtype=np.uint8)

    s = tile_size // downsample

    for Y in range(0, H, tile_size):
        line = result.read_region(
            location=(0, Y),
            level=0,
            size=(W, tile_size),
        )[:, :, 0]
        line = colors[line]
        line = cv2.resize(line, dsize=(w, s))
        z = Y // downsample
        visual[z: z+s, :, :] = line

    visual = cv2.cvtColor(src=visual, code=cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, visual)
