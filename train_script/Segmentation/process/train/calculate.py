import sys

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from basic import config, join
from utils.table.my_table import Table
from utils import Timer, TorchMerger
from dataset import Trainset
from .draw import draw

# 分类名称列
CLASS_NAMES = [
    'bg',  # 背景，值序：0，染色：浅灰
    'tm1',  # 浸润性肿瘤，值序：1，染色：亮红
    'st1',  # 肿瘤相关间质，值序：2，染色：亮黄
    'tm2',  # 原位肿瘤，值序：3，染色：暗红
    'nm',  # 健康腺体，值序：4，染色：深绿
    'nc',  # 非原位坏死，值序：5，染色：纯黑
    'st2',  # 炎症间质，值序：6，染色：橙黄
    'rs',  # 其它组织，值序：7，染色：纯白
]


def do_calculate(model: torch.nn.Module, data_dict: dict, T: Timer = None, visual: bool = False, visual_root: str = None):
    # 存储表
    table = Table(list(data_dict.keys()), dtype=list, key_sep='#', k_v_sep=':')
    with torch.no_grad():
        model.eval()
        for data_key, dataset in data_dict.items():
            if T: T.track(f' -> predicting {data_key}')
            # 获得预测结果
            argmaxed = predict(model, data_key, dataset)
            # 打印预测结果 -> 目录结合 keyset 确定
            if visual:
                draw(argmaxed, data_key=data_key, dataset=dataset, root=visual_root)
            # 统计样本评估的原始结果
            table[data_key] = calculate_single(argmaxed, data_key, dataset)
    return table


def calculate_single(argmaxed: np.ndarray, data_key: str, dataset: Trainset):
    l, u, r, d = dataset.loaders[data_key].box
    label = dataset.loaders[data_key].label({
        'x': (l + r) // 2,
        'y': (u + d) // 2,
        'w': (r - l),
        'h': (d - u),
        'degree': 0,
        'scaling': 1,
    })
    n = len(CLASS_NAMES)
    result = np.empty(shape=(n, n), dtype=int)
    for i in range(n):
        gt = label == i
        for j in range(n):
            pr = argmaxed == j
            # result[i, j] 表示 gt == i 且 pr == j 的像素的个数，因此：
            # result.sum(axis=1) 表示 gt 的像素数
            # result.sum(axis=0) 表示 pr 的像素数
            # result.diagonal() 表示 tp 的像素数
            result[i, j] = (gt & pr).sum()
    return result.tolist()


def predict(model: torch.nn.Module, data_key: str, dataset: Trainset):
    loader = DataLoader(
        dataset=dataset,
        batch_size=config['train.batch_size'],
        num_workers=config["train.num_workers"]
    )
    l, u, r, d = dataset.loaders[data_key].box
    with TorchMerger(class_num=8, kernel_size=512, kernel_steep=6).with_shape(W=r - l, H=d - u) as merger:
        with tqdm(loader, desc='valid', file=sys.stdout, disable=True) as iterator:
            for images, grids in iterator:
                images = images.to('cuda:0')
                predicts = model(images)
                merger.set(predicts, grids)
                del images, predicts, grids
            merged = merger.tail()
    del merger, loader
    return np.argmax(merged, axis=2)
