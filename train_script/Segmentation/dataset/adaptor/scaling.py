import re
import numpy as np


def trans_scaling(cfg: str) -> dict:
    mt = re.compile(r'(\w+)\[([\d,\.\s\-]+)\]')
    fd = re.compile(r'\s*([\d\.\-]+)\s*,?\s*')
    p = mt.match(cfg)
    assert p is not None, 'Pattern format not allowed!'
    head, indexes = p.groups()
    indexes = fd.findall(indexes)
    indexes = list(map(float, indexes))
    return {
        'random': head == 'random',
        'values': indexes,
    }


def get_scaling_rates(random: bool, values: list):
    if not random:
        return values
    # 生成随机缩放
    assert len(values) >= 2, 'Can not translate the random-scaling-indexes!'
    result = []
    for start, end in zip(values[:-1], values[1:]):
        result.append(np.random.random() * (end - start) + start)
    return result


if __name__ == '__main__':
    r = trans_scaling('random[1, 2]')
    print(r)
    t = get_scaling_rates(**r)
    print(t)
