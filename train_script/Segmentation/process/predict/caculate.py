import json
from typing import List
import torch
import numpy as np

from basic import source, join
from dataset import Trainset as Dataset
from utils import Timer, Table
from .predict import predict
from .draw import draw

# [255, 255, 255],  # index 0 -> undefined - Pure white
# [120, 120, 120],  # index 1 -> background - Light gray
# [220, 160, 0],  # index 2 -> stroma - Yellow orange
# [0, 0, 255],  # index 3 -> normal - Light blue
# [120, 0, 0],  # index 4 -> tumor - Dark red
# [0, 0, 0],  # index 5 -> necrosis - Pure black
# [0, 180, 0],  # index 6 -> Vessel - Dark green
# [40, 0, 40],  # index 7 -> IMMUNE CELLS - Dark purple
# 分类名称列
CLASS_NAMES = [
    'ud',   # undefined，无定义，值序：0，染色：纯白
    'bg',   # background，背景，值序：1，染色：浅灰
    'st',   # stroma，间质，值序：2，染色：橘黄
    'nm',   # normal，正常，值序：3，染色：纯蓝
    'tm',   # tumor，肿瘤，值序：4，染色：暗红
    'nc',   # necrosis，坏死，值序：5，染色：纯黑
    'vs',   # vessel，血管，值序：6，染色：暗绿
    'ic',   # immune cells，淋巴，值序：7，染色：暗紫
]


def do_calculate(model_name: str, root: str, T: Timer = None, visual: bool = False):
    # 数据集
    if T: T.track(' -> initializing dataset')
    dataset_dict = {
        name: Dataset(
            names=[name],
            length=-1,
            cropper_type='simple',
            shuffle=False,
            transfer={'level': 'off'},
            return_label=False,
            return_subset=False,
            return_pos=True,
            cropper_rotate='list[0]',
            cropper_scaling='list[1]',
        # ) for name in ['3|157957_637888001|2', '3|157957_637888001|1']
        # ) for name in source['group']['dev_test']
        ) for name in source['group']['available']
    }
    # 图集分包信息
    key_set = {key: 'train' for key in (
        source['group']['train'] + source['group']['test'] + [
            'TCGA-AC-A2QH-01Z-00-DX1.00B8BFFF-F1E2-4F99-A969-8DD7EE4F8E0B_[21089, 8432, 24106, 11383]',
            'TCGA-AC-A2QJ-01Z-00-DX1.48C303BB-5A23-4037-BD28-77629A8CD9DA_[18312, 4338, 19595, 7141]',
            'TCGA-EW-A1OW-01Z-00-DX1.97888686-EBB6-4B13-AB5D-452F475E865B_[18054, 24843, 22167, 29264]',
            'TCGA-GM-A2DF-01Z-00-DX1.CD0BE6D7-2DB3-4193-84CC-F9BE7BF18CC2_[25322, 21890, 27778, 24293]',
            'TCGA-EW-A1OV-01Z-00-DX1.93698123-5B34-4163-848B-2D75A5F7B001_[63016, 32569, 66793, 37935]',
            'TCGA-EW-A1P7-01Z-00-DX1.97575C9F-C318-45A5-A4B7-1A902B93FA3F_[4906, 23200, 6692, 25690]',
        ]
    )}
    key_set_2 = {key: 'valid' for key in source['group']['valid'] if key not in [
            'TCGA-AC-A2QH-01Z-00-DX1.00B8BFFF-F1E2-4F99-A969-8DD7EE4F8E0B_[21089, 8432, 24106, 11383]',
            'TCGA-AC-A2QJ-01Z-00-DX1.48C303BB-5A23-4037-BD28-77629A8CD9DA_[18312, 4338, 19595, 7141]',
            'TCGA-EW-A1OW-01Z-00-DX1.97888686-EBB6-4B13-AB5D-452F475E865B_[18054, 24843, 22167, 29264]',
            'TCGA-GM-A2DF-01Z-00-DX1.CD0BE6D7-2DB3-4193-84CC-F9BE7BF18CC2_[25322, 21890, 27778, 24293]',
            'TCGA-EW-A1OV-01Z-00-DX1.93698123-5B34-4163-848B-2D75A5F7B001_[63016, 32569, 66793, 37935]',
            'TCGA-EW-A1P7-01Z-00-DX1.97575C9F-C318-45A5-A4B7-1A902B93FA3F_[4906, 23200, 6692, 25690]',
        ]}
    key_set.update(key_set_2)
    # 确认数据集的合法性
    for data_key in dataset_dict.keys():
        assert data_key in key_set, f'something goes wrong -> {data_key}'
    # 统计表单：
    # axis 1 -> 模型名称
    # axis 2 -> 图片名称
    # axis 3 -> 评估矩阵 -> 标签端
    # axis 4 -> 评估矩阵 -> 预测端
    # 元素值类型均为 int64，表示统计像素数
    table = Table(
        list(dataset_dict.keys()), CLASS_NAMES, CLASS_NAMES,
        dtype=np.int64, key_sep='#', k_v_sep=':')
    with torch.no_grad():
        if T: T.track(f' -> initializing model -> {model_name}')
        m = torch.load(join(root, f'{model_name}.pth'))
        m = m.eval()
        for data_key, dataset in dataset_dict.items():
            # if key_set[data_key] != 'train': continue
            if T: T.track(f' -> predicting {data_key}')
            # 获得预测结果
            argmaxed = predict(m, data_key, dataset)
            # 打印预测结果 -> 目录结合 keyset 确定
            if visual:
                draw(argmaxed, data_key, dataset, root=join(root, model_name, key_set[data_key]))
            # 统计样本评估的原始结果
            table[data_key, :, :] = calculate_single(argmaxed, data_key, dataset)

    # 保存原始结果
    table.save(path=join(root, model_name, 'calculate.txt'))
    # 保存数据集分组信息
    key_set = json.dumps(key_set, indent=4)
    with open(join(root, 'key_set.txt'), 'w') as f:
        f.writelines(key_set)


def calculate_single(argmaxed: np.ndarray, data_key: str, dataset: Dataset):
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
    return result
