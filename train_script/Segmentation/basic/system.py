import sys
from .toolkit import Dict


# python 输入参数

SysArgs = Dict()

for arg in sys.argv:
    if arg and arg.count('=') == 1:
        key, val = arg.split('=')
        SysArgs[key] = val
