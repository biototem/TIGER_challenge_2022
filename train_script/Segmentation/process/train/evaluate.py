import json
from typing import Tuple
import numpy as np

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


def do_evaluate(data_table: Table):
    metrics = ['wdice', 'mdice', 'dice134', 'score', 'dice1', 'dice26', *(f'dice-on-{c}' for c in CLASS_NAMES)]
    # 以下开始进入评测流程
    score_table = Table(
        data_table.keys[0] + ['all'], metrics,
        k_v_sep=data_table.k_v_sep, key_sep=data_table.key_sep,
        dtype=float
    )
    # 计算单图评分
    for data_key in data_table.keys[0]:
        # 将统计矩阵还原为统计表
        matrix_table = Table(
            CLASS_NAMES,
            CLASS_NAMES,
            data=np.array(data_table[data_key]),
            dtype=np.int64,
        )
        # 开始打分
        dice, dice26, dice134, ms, ws = evaluate_dice(matrix_table=matrix_table)
        assert sum(ms) > 1, f'此图片不含有效标签，请酌情剔除！{data_key}'
        # wdice 是类间权重均衡 dice
        score_table[data_key, 'wdice'] = sum([d * g for d, g in zip(dice, ws)]) / sum(ws)
        score_table[data_key, 'mdice'] = sum([d * g for d, g in zip(dice, ms)]) / sum(ms)
        for i, c in enumerate(CLASS_NAMES):
            score_table[data_key, f'dice-on-{c}'] = dice[i]
        score_table[data_key, 'score'] = (dice[1] + dice26) / 2
        score_table[data_key, 'dice1'] = dice[1]
        score_table[data_key, 'dice26'] = dice26
        score_table[data_key, 'dice134'] = dice134

    # 计算组评分
    # 获得统计矩阵 -> 直接在 data_key 这个维度上求和即可
    matrix = sum([np.array(m, dtype=np.int64) for m in data_table.data])
    # 将统计矩阵还原为统计表
    matrix_table = Table(
        CLASS_NAMES,
        CLASS_NAMES,
        data=matrix,
    )
    # 开始打分
    dice, dice26, dice134, ms, ws = evaluate_dice(matrix_table=matrix_table)
    # 存储组评分 -> 组评分无需过滤死线数据
    score_table['all', 'wdice'] = sum([d * g for d, g in zip(dice, ws)]) / sum(ws)
    score_table['all', 'mdice'] = sum([d * g for d, g in zip(dice, ms)]) / sum(ms)
    for i, c in enumerate(CLASS_NAMES):
        score_table['all', f'dice-on-{c}'] = dice[i]
    score_table['all', 'score'] = (dice[1] + dice26) / 2
    score_table['all', 'dice1'] = dice[1]
    score_table['all', 'dice26'] = dice26
    score_table['all', 'dice134'] = dice134
    return score_table


def evaluate_dice(matrix_table: Table) -> Tuple[list, float, dict, dict]:
    # 评估矩阵的 numpy 数组
    matrix = matrix_table.data
    # 根据业务需求， label == 0 的标签不参与评测，因而 label == 0 的 pr 应当从评分中剔除
    pr = [int(matrix[1:, i].sum()) for i in range(8)]
    gt = [int(matrix[i, :].sum()) for i in range(8)]
    tp = [int(matrix[i, i]) for i in range(8)]
    dice = [2 * t / (p + g + 1e-17) for t, p, g in zip(tp, pr, gt)]

    # dice26 是混淆 stroma1、stroma2 的计分方式，此方式为赛方要求
    pr26 = int(pr[2] + pr[6])
    gt26 = int(gt[2] + gt[6])
    tp26 = int(matrix_table[(2, 6), (2, 6)].data.sum())
    dice26 = 2 * tp26 / (pr26 + gt26 + 1e-17)

    # dice134 是混淆 tumor1、tumor2、normal 的计分方式，此方式为徐哥要求
    pr134 = int(pr[1] + pr[3] + pr[4])
    gt134 = int(gt[1] + gt[3] + gt[4])
    tp134 = int(matrix_table[(1, 3, 4), (1, 3, 4)].data.sum())
    dice134 = 2 * tp134 / (pr134 + gt134 + 1e-17)

    # ms 表示当前图片或统计中的有效标签，其中 label == 0 始终无效
    ms = [g > 0 for g in gt]
    ms[0] = False

    # ws 表示当前图片的标签有效性权重，其中 -0.003465735 = np.log(0.5) / 200
    # 即：gt 像素数每增加 200，其”无效性“减半
    # 同时， label == 0 始终无效
    # ws = [1 - np.exp(-0.003465735 * g) for g in gt]
    ws = [1 - np.exp(-0.00069 * g) for g in gt]
    ws[0] = 0

    return dice, dice26, dice134, ms, ws
