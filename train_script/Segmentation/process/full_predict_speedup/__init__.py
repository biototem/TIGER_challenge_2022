import os
import torch

from basic import join
from utils import Timer
from .predictor import predict
from .visual import predict2visual
from .asap_slide import Slide as Reader, Writer


__all__ = ['do_full_predict_speedup']


def do_full_predict_speedup(
        model_name: str,
        image_path: str,
        mask_path: str,
        name: str,
        T: Timer = None,
        visual: bool = False,
        skip=True,
):
    KERNEL_SIZE = 512
    TILE_SIZE = 256
    TILE_STEP = 256
    SPACING = 0.5
    CHANNELS = 8
    DEVICE = 'cuda:0'
    DOWNSAMPLE = 16
    BATCH_SIZE = 16

    # ROOT 是预测输出目录，将来会配到 CONFIG 里
    ROOT = '../output/full_speedup/wsitils'
    # 预测结果有二：label.tif，visual.png
    os.makedirs(join(ROOT, 'label'), exist_ok=True)
    os.makedirs(join(ROOT, 'visual'), exist_ok=True)

    predict_path = join(ROOT, 'label', f'{name}.tif')
    if skip and os.path.exists(predict_path):
        return

    # 加载模型权重
    model = torch.load(model_name)
    model.eval()

    # 加载图片
    image = Reader(svs_file=image_path, level=0)

    # 该项目有 mask 图
    mask = Reader(svs_file=mask_path, level=0)

    # 配置输出信息
    dimensions = image.getDimensions()
    with torch.no_grad():
        with Writer(
            output_path=predict_path,
            tile_size=TILE_SIZE,
            dimensions=dimensions,
            spacing=SPACING,
        ) as writer:
            # 逐行预测与融合
            predict(
                model=model,                # 模型文件
                image=image,                # 输入流
                mask=mask,                  # 辅助输入流
                writer=writer,              # 输出流
                dimensions=dimensions,      # 维度数
                tile_size=TILE_SIZE,        # 写图核尺寸
                tile_step=TILE_STEP,        # 融合步长(tile_size 的整数倍)
                ksize=KERNEL_SIZE,          # 预测核尺寸
                channel=CHANNELS,           # 输出通道数
                batch_size=BATCH_SIZE,
                device=DEVICE,              # 预测所用容器
            )

    # 肺癌项目不需要后处理 -> 若需，在这里继续插入新流程

    # 可视化染色
    visual_path = join(ROOT, 'visual', f'{name}.png')
    result = Reader(svs_file=predict_path, level=0)
    predict2visual(result=result, output_path=visual_path, tile_size=TILE_SIZE, downsample=DOWNSAMPLE)
