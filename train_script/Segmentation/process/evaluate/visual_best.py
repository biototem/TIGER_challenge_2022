import json
import matplotlib.pyplot as plt

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


def visual(root: str):
    # 加载评分表
    score_table = Table.load(path=join(root, 'evaluate.txt'), dtype=object)
    # 加载分组信息
    with open(join(root, 'key_set.txt'), 'r') as f:
        key_set = f.read()
        key_set = json.loads(key_set)

    # 模型名
    model_names = score_table.keys[0]
    # 图片 ID
    data_keys = score_table.keys[1]
    # 评价 ID
    metric_keys = score_table.keys[2]

    idx = 0
    # plt.legend()
    # 逐评价指标
    for metric in metric_keys:
        # if metric not in [
        #     'mdice', 'score', 'wdice',
        #     'dice1', 'dice134', 'dice26',
        # ]: continue
        if metric not in [
            'score',
        ]: continue
        # idx += 1
        # plt.subplot(2, 3, idx)
        # 逐分组
        for group, color in zip(['train', 'valid'], ['g', 'y', 'r']):
            ys = score_table[model_names, group, metric].data.tolist()
            xs = [i + 1 for i in range(len(model_names))]
            plt.title(f'{metric}')
            plt.plot(xs, ys, color=color, label=group)
        plt.yticks([i / 20 for i in range(21)])
    plt.show()

    # idx = 0
    # plt.legend()
    # # 逐评价指标
    # for j, c in enumerate(CLASS_NAMES):
    #     idx += 1
    #     plt.subplot(2, 4, idx)
    #     # 逐分组
    #     for group, color in zip(['train', 'valid', 'test'], ['g', 'y', 'r']):
    #         ys = score_table[model_names, group, 'dice'].data.tolist()
    #         ys = [y[j] for y in ys]
    #         xs = [i + 1 for i in range(len(model_names))]
    #         plt.title(f'{c}')
    #         plt.plot(xs, ys, color=color, label=group)
    #     plt.yticks([i / 10 for i in range(11)])
    # plt.show()
