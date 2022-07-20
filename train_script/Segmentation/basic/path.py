import os, sys
from .system import SysArgs
from .toolkit import Dict

sep = os.path.sep
rpx = {'/': '\\', '\\': '/'}[sep]


# join 是 os.path.join 的一个增强缩写封装
def join(path: str, *more_paths: str):
    if path.startswith('~'):
        path = WORKSPACE + path[1:]
    path = os.path.join(path, *more_paths)
    return path.replace(rpx, sep)


# 默认 workspace 为运行目录
WORKSPACE = os.path.abspath(SysArgs['workspace'] or os.getcwd())
# 默认 配置文件 在根目录下
CONFIG_PATH = join(SysArgs['config'] or './CONFIG.yaml')


class Config(Dict):
    def __init__(self, dictionary: dict):
        super().__init__(dictionary)

    def __getitem__(self, items: str):
        if not items:
            return None
        target = dict(self)
        for item in items.split('.'):
            if not isinstance(target, dict) or item not in target:
                return None
            target = target[item]
        if isinstance(target, str):
            return join(target)
        return target
