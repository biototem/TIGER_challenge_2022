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

    # 逐模型
    for model_name in model_names:
        # 逐评价指标
        for metric in metric_keys:
            # 逐分组
            for divide in ['train', 'valid', 'test']:
                keys = [k for k in data_keys[:-3] if key_set[k] == divide]
                datas = score_table[model_name, keys, metric]
                data = score_table[model_name, divide, metric]
                if metric == 'dice':
                    datas = datas.data.tolist()
                    datas = [sum(d[j] for d in datas) / len(datas) for j in range(8)]
                    # for i, class_name in enumerate(CLASS_NAMES):
                    #     print(f'{model_name}-{divide}-mean-image-dice-on-{class_name} = {datas[i]}')
                    for i, class_name in enumerate(CLASS_NAMES):
                        print(f'{model_name}-{divide}-pixes-dice-on-{class_name} = {data[i]}')
                elif metric == 'weight':
                    continue
                else:
                    datas = datas.data.mean(axis=0)
                    print(f'{model_name}-{divide}-mean-{metric} = {datas}')
                    print(f'{model_name}-{divide}-pixes-{metric} = {data}')
