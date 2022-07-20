import math
import os
from typing import Tuple, List

import torch
import numpy as np
from tqdm import tqdm

from basic import join
from utils import Timer, Region, Shape
from .predictor import predict
from .visual import predict2visual
from .redrawer import redraw
from .asap_slide import Slide as Reader, Writer
from .wt import wt_predict, wt_interface_trans_img

__all__ = ['do_submit_predict_speedup']


def do_submit_predict_speedup(
        model_name: str,
        image_path: str,
        mask_path: str,
        name: str,
        T: Timer = None,
        visual: bool = False
):

    KERNEL_SIZE = 512
    TILE_SIZE = 128
    TILE_STEP = 256
    SPACING = 0.5
    CHANNELS = 8
    DEVICE = 'cuda:0'
    DOWNSAMPLE = 16
    BATCH_SIZE = 1

    # ROOT 是预测输出目录，将来会配到 CONFIG 里
    ROOT = '~/output/submit_speedup'
    # 预测结果有二：label.tif，visual.png
    os.makedirs(join(ROOT, 'label'), exist_ok=True)
    os.makedirs(join(ROOT, 'visual'), exist_ok=True)

    # 加载模型权重
    model = torch.load(model_name)
    model.eval()

    # 加载图片
    image = Reader(svs_file=image_path, level=0)

    # 该项目有 mask 图
    mask = Reader(svs_file=mask_path, level=0)

    dimensions = image.getDimensions()

    # 逐行预测与融合
    # TODO: 警告！ 下列代码有错误！ 暂时无法确定错误原因， 可对比以下两张图片帮逐确定原因
    # /media/totem_disk/totem/jizheng/breast_competation/output/submit_speedup/label/130S_wrong.tif
    # /media/totem_disk/totem/jizheng/breast_competation/output/submit_speedup/label/130S_right.tif
    with torch.no_grad():
        character_matrix, adjacencies, contours, stroma = predict(
            model=model,  # 模型文件
            image=image,  # 输入流
            mask=mask,  # 辅助输入流
            dimensions=dimensions,  # 维度数
            tile_size=TILE_SIZE,  # 写图核尺寸
            tile_step=TILE_STEP,  # 融合步长(tile_size 的整数倍)
            ksize=KERNEL_SIZE,  # 预测核尺寸
            channel=CHANNELS,  # 输出通道数
            batch_size=BATCH_SIZE,
            device=DEVICE,  # 预测所用容器
        )
    # TODO： END

    # TODO: 通过这套代码生成的图是正确的，可以确定后续流程并没有问题
    # predict_path = '/media/totem_disk/totem/jizheng/breast_competation/output/full_speedup/label/130S.tif'
    # argmaxed = Reader(svs_file=predict_path, level=0)
    # character_matrix, adjacencies, contours, stroma = find_contours(
    #     argmaxed=argmaxed,      # 输入流
    #     image=image,           # 辅助输入流
    #     tile_size=TILE_SIZE,
    #     dimensions=dimensions,  # 维度数
    # )
    # TODO： END

    # step 1 -> 建立特征阵列与轮廓的关系
    if T: T.track(f' -> post-process: map indexes')
    contours = contours.sep_out()
    # step 2 -> 用文泰模型预测特征阵列
    if T: T.track(f' -> post-process: predict by wt')
    if not adjacencies:
        assert not contours
        attentions = []
    else:
        adjacency_xs, adjacency_ys = zip(*adjacencies)
        character_list = character_matrix[adjacency_ys, adjacency_xs]
        attention_list = wt_predict(character_list.tolist())
        attention_matrix = np.zeros_like(character_matrix, dtype=np.float32)
        attention_matrix[adjacency_ys, adjacency_xs] = attention_list
        # step 3 -> 标定轮廓性质
        if T: T.track(f' -> post-process: mark contour if positive')
        attentions = build_attentions(attention_matrix, dimensions, contours, TILE_SIZE)

    # 配置输出信息
    predict_path = join(ROOT, 'label', f'{name}.tif')
    with Writer(
        output_path=predict_path,
        tile_size=TILE_SIZE,
        dimensions=dimensions,
        spacing=SPACING,
    ) as writer:
        redraw(
            contours=contours,      # 轮廓标记
            attentions=attentions,  # 轮廓归类
            stromas=stroma.sep_out(),
            writer=writer,          # 输出流
            dimensions=dimensions,  # 维度数
            tile_size=TILE_SIZE,    # 写图核尺寸
        )

    # 可视化染色
    # visual_path = join(ROOT, 'visual', f'{name}.png')
    # result = Reader(svs_file=predict_path, level=0)
    # predict2visual(result=result, output_path=visual_path, tile_size=TILE_SIZE, downsample=DOWNSAMPLE)


def build_attentions(attention_matrix: np.ndarray, dimensions: Tuple[int, int], contours: List[Shape], tile_size: int):

    attentions = []

    W, H = dimensions
    w, h = math.ceil(W / tile_size), math.ceil(H / tile_size)

    for index, contour in enumerate(contours):
        l, u, r, d = contour.bounds
        l = math.floor(l / tile_size)
        u = math.floor(u / tile_size)
        r = math.ceil(r / tile_size)
        d = math.ceil(d / tile_size)
        attention_list = []
        for i, j in [(_i, _j) for _i in range(l, r) for _j in range(u, d)]:
            region = Region(left=i, up=j, right=i+1, down=j+1) * tile_size
            if not region.is_joint(contour):
                continue
            attention = attention_matrix[j, i]
            attention_list.append(attention)

        assert len(attention_list) > 0, '??????'
        attention = sum(attention_list) / len(attention_list)
        attentions.append(attention > 0.5)
    return attentions


def find_contours(
    argmaxed: Reader,
    image: Reader,
    tile_size: int,
    dimensions: Tuple[int, int],
) -> Shape:

    from utils import ComplexMultiPolygon
    from .predictor import find_contours as find_contour

    W, H = dimensions
    # 图片的宽高对 tile_size 整数倍向上取整
    W = math.ceil(W / tile_size) * tile_size
    H = math.ceil(H / tile_size) * tile_size

    # contours = ComplexMultiPolygon.EMPTY
    contours = []
    stromas = []

    # wt_matrix
    character_matrix = np.empty(shape=(math.ceil(H / tile_size), math.ceil(W / tile_size)), dtype=object)
    adjacencies = []

    # 逐片迭代
    for j, y in tqdm([
        (_j, _y) for _j, _y in enumerate(range(0, H, tile_size))
    ], desc='predicting'):
        # 获得预测结果
        temp = argmaxed.read_region(
            location=(0, y),
            level=0,
            size=(W, tile_size),
        )[:, :, 0]

        # make character
        counts = (temp == 1) | (temp == 3) | (temp == 4)
        for x in range(0, W, tile_size):
            tile = counts[:, x: x+tile_size].astype(np.float32)
            if tile.any() and tile.mean() > 0.05:
                patch = image.read_region(
                    location=(x, y),
                    level=0,
                    size=(tile_size, tile_size),
                )
                character_matrix[y // tile_size, x // tile_size] = wt_interface_trans_img(torch.tensor(patch))
                adjacencies.append([x // tile_size, y // tile_size])

        # find_contour
        counts = (temp == 1) | (temp == 3) | (temp == 4)
        contour = find_contour(counts) + (0, y)
        contour = contour.buffer(distance=1)
        contour = contour.simplify(tolerance=1)
        contours.extend(contour.sep_out())

        counts = (temp == 2) | (temp == 6)
        stroma = find_contour(counts) + (0, y)
        stroma = stroma.buffer(distance=1)
        stroma = stroma.simplify(tolerance=1)
        stromas.extend(stroma.sep_out())

    contours = ComplexMultiPolygon(singles=contours)
    contours = contours.standard()
    contours = contours.buffer(distance=-1)
    contours = contours.simplify(tolerance=1)

    stromas = ComplexMultiPolygon(singles=stromas)
    stromas = stromas.standard()
    stromas = stromas.buffer(distance=-1)
    stromas = stromas.simplify(tolerance=1)

    return character_matrix, adjacencies, contours, stromas
