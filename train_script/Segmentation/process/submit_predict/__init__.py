import math
import os
import numpy as np
from typing import List, Tuple
import torch

from basic import join
from utils import Timer, Shape, Region
from .predictor import predict
from .asap_slide import Slide as Reader, Writer
from .redrawer import redraw
from .visual import predict2visual


__all__ = ['do_submit_predict']

from .wt_interface import wt_predict


def do_submit_predict(
        name: str,
        model_path: str,
        image_path: str,
        mask_path: str,
        T: Timer = None,
        visual: bool = False
):
    if T: T.track(f' -> start submit predicting {name}')
    KERNEL_SIZE = 512
    TILE_SIZE = 128
    TILE_STEP = 256
    SPACING = 0.5
    CHANNELS = 8
    DEVICE = 'cuda:0'
    DOWNSAMPLE = 32

    ROOT = join('~/output/submit')
    os.makedirs(join(ROOT, 'predict'), exist_ok=True)
    os.makedirs(join(ROOT, 'contour'), exist_ok=True)
    os.makedirs(join(ROOT, 'visual'), exist_ok=True)

    # 加载模型权重
    if T: T.track(f' -> loading model at path {model_path}')
    model = torch.load(model_path)
    model.eval()

    # 加载图片
    if T: T.track(f' -> loading image at path {image_path}')
    image = Reader(svs_file=image_path, level=0)

    # 该项目有 mask 图
    if T: T.track(f' -> loading mask at path {mask_path}')
    mask = Reader(svs_file=mask_path, level=0)

    # 配置输出信息
    dimensions = image.getDimensions()
    predict_path = join(ROOT, 'predict', f'{name}.tif')
    if T: T.track(f' -> output predict at path {predict_path}')
    with Writer(
        output_path=predict_path,
        tile_size=TILE_SIZE,
        dimensions=dimensions,
        spacing=SPACING,
    ) as writer:
        # 逐行预测与融合 -> 顺便获得特征阵列与轮廓数组
        character_matrix, contours = predict(
            model=model,                # 模型文件
            image=image,                # 输入流
            mask=mask,                  # 辅助输入流
            writer=writer,              # 输出流
            dimensions=dimensions,      # 维度数
            tile_size=TILE_SIZE,        # 写图核尺寸
            tile_step=TILE_STEP,        # 融合步长(tile_size 的整数倍)
            ksize=KERNEL_SIZE,          # 预测核尺寸
            channel=CHANNELS,           # 输出通道数
            device=DEVICE,              # 预测所用容器
            T=T and T.tab(),
        )
    # argmaxed = Reader(svs_file=predict_path, level=0)
    # contours = find_contours(
    #     argmaxed=argmaxed,      # 输入流
    #     mask=mask,              # 辅助输入流
    #     tile_size=TILE_SIZE,
    #     dimensions=dimensions,  # 维度数
    # )
    # character_matrix = np.empty(shape=(math.ceil(dimensions[1] / TILE_SIZE), math.ceil(dimensions[0] / TILE_SIZE)), dtype=object)

    # 肺癌项目后处理
    # step 1 -> 建立特征阵列与轮廓的关系
    if T: T.track(f' -> post-process: map indexes')
    contours = contours.sep_out()
    index_matrix = build_map(dimensions, contours, TILE_SIZE)
    # step 2 -> 用文泰模型预测特征阵列
    if T: T.track(f' -> post-process: predict by wt')
    attention_matrix = wt_predict(character_matrix)
    # step 3 -> 标定轮廓性质
    if T: T.track(f' -> post-process: mark contour if positive')
    attentions = build_attentions(attention_matrix, dimensions, contours, TILE_SIZE)

    # step 4 -> 轮廓反绘
    contour_path = join(ROOT, 'contour', f'{name}.tif')
    if T: T.track(f' -> post-process: redraw contours to predict-result at {contour_path}')
    argmaxed = Reader(svs_file=predict_path, level=0)
    with Writer(
        output_path=contour_path,
        tile_size=TILE_SIZE,
        dimensions=dimensions,
        spacing=SPACING,
    ) as writer:
        redraw(
            argmaxed=argmaxed,      # 输入流
            mask=mask,              # 辅助输入流
            contours=contours,      # 轮廓标记
            index_matrix=index_matrix,  # 序数矩阵
            attentions=attentions,  # 轮廓归类
            writer=writer,          # 输出流
            dimensions=dimensions,  # 维度数
            tile_size=TILE_SIZE,    # 写图核尺寸
        )

    # 可视化染色
    visual_path = join(ROOT, 'visual', f'{name}.png')
    if T: T.track(f' -> visualizing at {visual_path}')
    result = Reader(svs_file=contour_path, level=0)
    predict2visual(result=result, output_path=visual_path, tile_size=TILE_SIZE, downsample=DOWNSAMPLE)


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


def build_map(dimensions: Tuple[int, int], contours: List[Shape], tile_size: int):
    W, H = dimensions
    w, h = math.ceil(W / tile_size), math.ceil(H / tile_size)

    matrix = np.empty(shape=(h, w), dtype=list)
    for index, contour in enumerate(contours):
        l, u, r, d = contour.bounds
        l = math.floor(l / tile_size)
        u = math.floor(u / tile_size)
        r = math.ceil(r / tile_size)
        d = math.ceil(d / tile_size)
        for i, j in [(_i, _j) for _i in range(l, r) for _j in range(u, d)]:
            region = Region(left=i, up=j, right=i+1, down=j+1) * tile_size
            if region.is_joint(contour):
                if matrix[j, i] is None:
                    matrix[j, i] = []
                matrix[j, i].append(index)
    return matrix


def find_contours(
    argmaxed: Reader,
    mask: Reader,
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

        contour = find_contour(temp[:, :, 0]) + (0, y)
        # contour = contour.smooth(distance=-5)
        contour = contour.simplify(tolerance=5)
        contours.extend(contour.sep_out())

    contours = ComplexMultiPolygon(singles=contours)
    contours = contours.standard()
    contours = contours.smooth(distance=-15)
    contours = contours.simplify(tolerance=5)

    return contours
