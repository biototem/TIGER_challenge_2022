import random
import json
import re

from basic import join
from utils import Assert


def build_group():
    # 本次预购建流程依赖手调的数据全集（排除了部分脏数据、错数据，并引用了部分 cells 数据）
    Assert.file_exist(join('~/resource/manual/image_uses.json'))
    # 以下为具体的分组规则：
    # available: 有效数据 (available = train + valid + test)
    # blocked： 无效数据 (依据 tag 做进一步区分)
    # train: 训练数据
    # valid: 验证数据
    # test: 测试数据
    # dev_test: 开发测试数据（在 available 范围内任意的手动指定即可）

    with open(join('~/resource/manual/image_uses.json')) as f:
        all_datas = json.load(f)

    # 首先分离有效 / 无效
    available = dict((k, v) for k, v in all_datas.items() if v['available'])
    blocked = set(k for k, v in all_datas.items() if not v['available'])

    # 对 folder 1 按机构名划分 train / valid
    train_rule = ['A7', 'A2', 'AR', 'BH', 'OL', 'E2', 'C8', 'A8', 'D8', 'AQ', 'GI']
    train_rule = [f'^TCGA\\-{name}.*$' for name in train_rule]
    train_rule = '|'.join(train_rule)
    train_rule = re.compile(train_rule)
    train_folder_1 = set(k for k, v in available.items() if v['folder'] == 1 and bool(train_rule.findall(k)))
    valid_folder_1 = set(k for k, v in available.items() if v['folder'] == 1 and not bool(train_rule.findall(k)))
    # folder 2 直接划入 valid
    valid_folder_2 = set(k for k, v in available.items() if v['folder'] == 2)
    # folder 3 随机划分 train / test
    available_folder_3 = set(k for k, v in available.items() if v['folder'] == 3)
    train_folder_3 = set(random.sample(available_folder_3, len(available_folder_3) // 2))
    test_folder_3 = available_folder_3 - train_folder_3

    # 整理数据集
    train = train_folder_1 | train_folder_3
    valid = valid_folder_1 | valid_folder_2
    test = test_folder_3

    # manual deal with data
    md = {
        'TCGA-AC-A2QH-01Z-00-DX1.00B8BFFF-F1E2-4F99-A969-8DD7EE4F8E0B_[21089, 8432, 24106, 11383]',
        'TCGA-AC-A2QJ-01Z-00-DX1.48C303BB-5A23-4037-BD28-77629A8CD9DA_[18312, 4338, 19595, 7141]',
        'TCGA-EW-A1OW-01Z-00-DX1.97888686-EBB6-4B13-AB5D-452F475E865B_[18054, 24843, 22167, 29264]',
        'TCGA-GM-A2DF-01Z-00-DX1.CD0BE6D7-2DB3-4193-84CC-F9BE7BF18CC2_[25322, 21890, 27778, 24293]',
        'TCGA-EW-A1OV-01Z-00-DX1.93698123-5B34-4163-848B-2D75A5F7B001_[63016, 32569, 66793, 37935]',
        'TCGA-EW-A1P7-01Z-00-DX1.97575C9F-C318-45A5-A4B7-1A902B93FA3F_[4906, 23200, 6692, 25690]',
    }
    train = train | test | md
    valid = valid - md
    test = []

    k1, k2 = next(zip(train, valid))
    dev_test = {k1, k2}

    return {
        'all': list(all_datas.keys()),
        'available': list(available.keys()),
        'blocked': list(blocked),
        'train': list(train),
        'valid': list(valid),
        'test': list(test),
        'dev_test': list(dev_test),
    }


def select(data: dict, rate: float, group_flag: str = None, already: list = None) -> dict:
    assert 0 <= rate <= 1, f'rate must be a float within [0, 1], found: {rate}'

    source = set(filter(lambda k: 'group' not in data[k] or not data[k]['group'], data.keys()))
    if already:
        for l in already:
            source -= l.keys()
    result = set(filter(lambda k: group_flag and 'group' in data[k] and data[k]['group'] == group_flag, data.keys()))

    n = int(len(data) * rate)

    # 图片数不足，一次全给
    if n >= len(source) + len(result):
        result |= source
    else:
        # 否则补足差额
        n -= len(result)
        result |= set(random.sample(source, n))
    return dict((k, data[k]) for k in result)
