import random
import json

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
    blocked = dict((k, v) for k, v in all_datas.items() if not v['available'])

    # 由于测试集固定占据了接近一半的图片数，因此训练集和验证集共同分享剩下那些
    train = select(data=available, rate=0.4, group_flag='train', already=[])
    valid = select(data=available, rate=1, group_flag='valid', already=[train])
    test = select(data=available, rate=1, group_flag='test', already=[train, valid])
    hard = dict((k, v) for k, v in all_datas.items() if v['tag'] == 'HARD')

    (k1, v1), (k2, v2), (k3, v3) = next(zip(train.items(), valid.items(), test.items()))
    dev_test = {k1: v1, k2: v2, k3: v3}

    return {
        'all': list(all_datas.keys()),
        'available': list(available.keys()),
        'blocked': list(blocked.keys()),
        'train': list(train.keys()),
        'valid': list(valid.keys()),
        'test': list(test.keys()),
        'hard': list(hard.keys()),
        'dev_test': list(dev_test.keys()),
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
