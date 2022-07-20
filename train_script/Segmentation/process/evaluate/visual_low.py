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


def visual(root: str, low: float = 0.8, names: list = None):
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

    # 逐模型
    for model_name in (names or model_names):
        # 逐评价指标
        # for metric in ['score', 'dice1', 'dice26']:
        for metric in ['score']:
            # 逐分组
            for divide in ['train', 'valid', 'test']:
                keys = [k for k in data_keys[:-3] if key_set[k] == divide]
                datas = [(key, score_table[model_name, key, metric]) for key in keys]
                datas = [w for w in datas if w[1] < low]
                datas.sort(key=lambda w: w[1])
                for k, v in datas:
                    print(f'{model_name}-{divide}-{metric} = {v}: {k}')
                # 逐图片
                # for key in keys:
                #     val = score_table[model_name, key, metric]
                #     if val <= low:
                #         print(f'{model_name}-{divide}-{metric} = {val}: {key}')
