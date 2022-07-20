from typing import List
import copy


def init_dict(keys: List[str], temp: dict):
    d = dict()
    for key in keys:
        d[key] = copy.deepcopy(temp)
    return d
