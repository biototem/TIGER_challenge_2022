import numpy as np
import json

from basic import join
from utils.table.my_table import Table

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


def visual(root: str, low: float = 0.8):
    # 加载统计信息
    data_table = Table.load(path=join(root, 'calculate.txt'), dtype=np.int64)
    # 加载分组信息
    with open(join(root, 'key_set.txt'), 'r') as f:
        key_set = f.read()
        key_set = json.loads(key_set)

    # 模型名
    model_names = data_table.keys[0]
    # 图片标签
    data_keys = data_table.keys[1]
    # 逐模型
    for model_name in model_names:
        # 计算组评分
        for group in ['train', 'valid', 'test']:
            # 获取分组信息
            group_keys = list(filter(lambda key: key_set[key] == group, data_keys))
            assert len(group_keys) >= 2, f'分组数量少于 2 是不可能的，请检查这里的 bug：{group_keys}'
            # 获得统计矩阵 -> 直接在 data_key 这个维度上求和即可
            matrix = data_table[model_name, group_keys].data.sum(axis=0)
            print(f'\n{model_name.replace("auto-save-epoch-", "2-2-")} -> {group}')
            print_matrix(matrix=matrix, row_names=CLASS_NAMES, col_names=CLASS_NAMES)
            # 将统计矩阵还原为统计表
            # matrix_table = Table(
            #     CLASS_NAMES,
            #     CLASS_NAMES,
            #     data=matrix,
            # )


def print_matrix(matrix: np.ndarray, row_names: list, col_names: list):
    # 段起始符
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    # 控制符
    n = 8
    print(' ' * n, end='')
    for name in col_names:
        print(f'%{n}s' % name, end='')
    print()
    for name, line in zip(row_names, matrix):
        print(f'%{n}s' % name, end='')
        for v in line:
            print(f'%{n}s' % ('%.0e' % v), end='')
        print()
    # 段终止符
    print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
