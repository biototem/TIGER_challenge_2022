import yaml

from .cuda_env import *
from .path import CONFIG_PATH, join, Config
from .toolkit import Dict, InterfaceProxy


__all__ = ['config', 'source', 'join', 'Dict', 'InterfaceProxy']


with open(CONFIG_PATH, "rb") as fp:
    config = yaml.full_load(fp)
    config = Config(config)

if os.path.exists(config['source.lib']):
    with open(config['source.lib'], 'rb') as fp:
        source = Dict(yaml.full_load(fp))
else:
    source = Dict()
