import yaml

from basic import config, source
from utils import Assert, Timer
from .group_manual import build_group
from .data import build_data


def do_build(T: Timer = None):
    Assert.not_none(source_lib=config['source.lib'])
    # 创建分组信息 -> 这些分组信息主要用来规定训练集、验证集、测试集
    if T: T.track(' -> building group')
    group_lib = build_group()
    # 创建数据信息 -> 这些数据信息主要用来搭建数据集
    if T: T.track(' -> building data')
    data_lib = build_data(visual=False, T=T and T.tab())
    # 生成预构建信息
    lib = {
        'group': group_lib,
        'data': data_lib,
    }
    if T: T.track(f' -> saving source_lib at {config["source.lib"]}')
    # 更新内存
    source.clear()
    source.update(lib)
    # 更新硬盘
    with open(config['source.lib'], mode='w', encoding='utf-8') as fp:
        yaml.dump(lib, fp)
