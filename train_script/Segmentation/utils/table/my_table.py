from typing import List, Iterable, Union, Tuple
from functools import reduce
import numpy as np
import re


class Table(object):
    """
    Table is a data-structure for multi-key-data caching in memory.
    It designed for score-table, which may use keys like 'model-name', 'dataset-type', 'evaluate-scope' and etc.
    Most possibly, it uses all those mentioned above.
    So, it allows to select like this:
    # Define a table
    table = Table(['ce', 'dice', 'dice(1,3)'], ['train', 'valid', 'test'], ['pixes', 'image', 'batch', 'patch'])
    # Assume the following list stands for ['ce-score', 'dice-score', 'dice(1, 3)-score']
    # in 'pixes' scope and 'train' dataset-type
    # Values type 1
    score_list_for_models = [0.5, 0.6, 0.7]
    # Values type 2
    score_list_for_models = {'ce': 0.5, 'dice': 0.6, 'dice(1,3)': 0.7}
    # Set method 1
    table[:, 'train', 'pixes'] = score_list_for_models
    # Set method 2
    table[('ce', 'dice', 'dice(1,3)'), 'train', 'pixes'] = score_list_for_models
    # Set method 3
    table[0:3, 'train', 'pixes'] = score_list_for_models
    # Set method 4
    table[(0,1,2), 'train', 'pixes'] = score_list_for_models
    # Get method 1 -> Table(['train', 'valid', 'test'], ['pixes', 'image', 'batch', 'patch'])
    table['ce', :, :]
    # Get method 2 -> Value
    table['ce', 'valid', 'image']
    """

    def __init__(self, *key_names_list: List[str],
                 dtype: type = object,
                 data: np.ndarray = None,
                 key_sep: str = '-',
                 k_v_sep: str = ': ',):
        assert key_names_list is not None and len(key_names_list) > 0, 'There must be at least one key_name in params.'
        assert sum(len(names) <= 1 for names in key_names_list) == 0, 'Shape eq 0 or 1 is meaningless!'
        self.key_sep = key_sep
        self.k_v_sep = k_v_sep
        self.keys = key_names_list
        self.idx = [{n: i for i, n in enumerate(key_names)} for key_names in key_names_list]
        self.data = np.empty(shape=list(len(key_names) for key_names in key_names_list), dtype=dtype) \
            if data is None else data

    def __getitem__(self, items):
        # 先将输入规范化为元组
        if not isinstance(items, tuple):
            items = items,
        # 然后补齐缺失轴
        if len(items) < len(self.keys):
            items = *items, *(slice(None) for _ in range(len(self.keys) - len(items)))
        # 然后逐项翻译
        selects = []
        for pos, item in enumerate(items):
            # 若 iterable：逐项翻译
            # 若 slice：按规则翻译
            # 翻译 str： 参照idx
            # 翻译 int： 直取
            if isinstance(item, int) or isinstance(item, str):
                select = self.trans(pos, item),
            elif isinstance(item, Iterable):
                select = self.trans_iter(pos, item)
            elif isinstance(item, slice):
                select = self.trans_slice(pos, item)
            else:
                raise KeyError(f'Not supported type: {type(item)}')
            assert bool(select), f'Select nothing is meaningless! Check pos[{pos}] val[{item}]'
            selects.append(list(select))
        grids = self.select(selects)
        selected_data = self.data[grids]
        if all(isinstance(g, int) for g in grids):
            # Only when every axis has only one index, return single value
            if self.data.dtype != object:
                selected_data = selected_data.item()
            return selected_data
        else:
            # Else return Table
            return self.sub_table(selects, selected_data)

    def trans_str(self, pos: int, item: str) -> tuple:
        return self.idx[pos][item]

    def trans(self, pos: int, item: Union[int, str]) -> int:
        if item is None: return None
        assert isinstance(item, int) or isinstance(item, str), KeyError(f'Not supported type: {type(item)}')
        if isinstance(item, int): return item
        assert item in self.idx[pos], KeyError(f'Key "{item}" not in key-index-{pos + 1}')
        return self.idx[pos][item]

    def trans_slice(self, pos: int, item: slice) -> tuple:
        start = self.trans(pos, item.start)
        stop = self.trans(pos, item.stop)
        step = self.trans(pos, item.step)
        start = 0 if start is None else start
        stop = len(self.idx[pos]) if stop is None else stop
        step = 1 if step is None else step
        return tuple(range(start, stop, step))

    def trans_iter(self, pos: int, items: Iterable) -> tuple:
        return tuple(self.trans(pos, item) for item in items)

    def select(self, indexes: List[Tuple[int]]) -> np.ndarray:
        # 若索引指定唯一元素，则按单元素索引元组格式返回坐标
        if all(len(index) == 1 for index in indexes):
            return tuple(index[0] for index in indexes)
        # 若维度数少于2，可以直接返回
        if len(indexes) < 2:
            return tuple(indexes)
        # 否则先生成网格矩阵
        grids = np.meshgrid(*indexes)
        # 第一个维度和第二个维度需要转置 -> 我也不知道为什么
        axis = list(range(len(indexes)))
        axis[0] = 1
        axis[1] = 0
        # 然后返回 grid
        return tuple(grid.transpose(axis) for grid in grids)

    def sub_table(self, selects: List[Tuple[int]], selected_data: np.ndarray):
        names_list = [[self.keys[axis][index] for index in select] for axis, select in enumerate(selects)]
        names_list = [names for names in names_list if len(names) > 1]
        selected_data = np.squeeze(selected_data)
        return Table(*names_list, data=selected_data)

    def __setitem__(self, items, value):
        # 先将输入规范化为元组
        if not isinstance(items, tuple):
            items = items,
        # 然后补齐缺失轴
        if len(items) < len(self.keys):
            items = *items, *(slice(None) for _ in range(len(self.keys) - len(items)))
        # 然后逐项翻译
        selects = []
        # 检测是否包含序列 -> 这关系到返回值的类型
        iterable_flag = False
        for pos, item in enumerate(items):
            # 若 iterable：逐项翻译
            # 若 slice：按规则翻译
            # 翻译 str： 参照idx
            # 翻译 int： 直取
            if isinstance(item, int) or isinstance(item, str):
                select = self.trans(pos, item),
            elif isinstance(item, Iterable):
                select = self.trans_iter(pos, item)
            elif isinstance(item, slice):
                select = self.trans_slice(pos, item)
            else:
                raise KeyError(f'Not supported type: {type(item)}')
            selects.append(list(select))
        grids = self.select(selects)
        self.data[grids] = value

    def __str__(self):
        return self.str(type_repr=repr)

    def str(self, type_repr: callable) -> str:
        # 以 axis = 0 为 temp
        joints = [(key, [index]) for index, key in enumerate(self.keys[0])]
        # 逐 axis 做笛卡尔积
        for new_keys in self.keys[1:]:
            new_joints = []
            # 遍历 joints
            for last_key, last_indexes in joints:
                # 遍历 keys
                for new_index, new_key in enumerate(new_keys):
                    new_key = f'{last_key}{self.key_sep}{new_key}'
                    new_indexes = last_indexes + [new_index]
                    new_joints.append((new_key, new_indexes))
            joints = new_joints
        keys = [key for key, _ in joints]
        vals = [type_repr(self.data[tuple(indexes)]) for _, indexes in joints]
        content = '\n\t'.join(f'{key}{self.k_v_sep}{val}' for key, val in zip(keys, vals))
        return '{\n\t' + content + '\n}'

    def save(self, path: str, type_repr: callable = repr):
        with open(path, 'w') as f:
            # 首行存： 键值分隔符
            f.write(f'k-v-sep[{self.k_v_sep}]\n')
            # 次行存： 键分隔符
            f.write(f'key-sep[{self.key_sep}]\n')
            # 第三行存维度数
            f.write(f'axis{[len(key) for key in self.keys]}\n')
            # 接下来 len(axis) 行存每个维度的名称
            for key in self.keys:
                f.write(f'{key}\n')
            # 最后存数据
            f.writelines(self.str(type_repr=type_repr))

    @staticmethod
    def load(path: str, dtype: type = object, type_loader: callable = eval):
        with open(path, 'r') as f:
            lines = f.readlines()
        # 首行读： 键值分隔符
        # LINE_TEMP: k-v-sep[ANY_WORD]
        k_v_sep = re.compile(r'^\s*k-v-sep\[(.*)\]\s*$').findall(lines[0])[0]
        # Next line need env-python3.9
        # k_v_sep = lines[0].strip().removeprefix('k-v-sep[').removesuffix(']')

        # 次行读： 键分隔符
        # LINE_TEMP: key-sep[ANY_WORD]
        key_sep = re.compile(r'^\s*key-sep\[(.*)\]\s*$').findall(lines[1])[0]
        # Next line need env-python3.9
        # k_v_sep = lines[0].strip().removeprefix('k-v-sep[').removesuffix(']')

        # 三行读： 维度数
        # LINE_TEMP: axis[N1, N2, N3, ...]
        axis = re.compile(r'^\s*axis(.*)\s*$').findall(lines[2])[0]
        axis = eval(axis)
        # Next line need env-python3.9
        # k_v_sep = lines[0].strip().removeprefix('k-v-sep[').removesuffix(']')

        # 接下来 len(axis) 行读： 每个维度的名称
        # LINE_TEMP: ['', '', '', ...]
        key_names = [eval(line) for line in lines[3: 3 + len(axis)]]

        # 剩下的是封印在大括号中的键值对
        lines = lines[3 + len(axis) + 1: -1]

        # 行数与 axis 笛卡尔积不一致的，果断报错
        assert len(lines) == reduce(int.__mul__, axis), f'Lines-length not eq axis! check {axis} -> {len(lines)}'
        # 现在，是时候加载数据了
        table = Table(
            *key_names,
            dtype=dtype,
            key_sep=key_sep,
            k_v_sep=k_v_sep,
        )
        for line in lines:
            keys, value = line.strip().split(k_v_sep)
            keys = keys.split(key_sep)
            value = type_loader(value)
            table[tuple(keys)] = value
        return table


# path = '/media/totem_disk/totem/jizheng/breast_competation/recycle/test/save.txt'
#
# table = Table(['a', 'b', 'c'], ['aa', 'bb'], ['aaa', 'bbb', 'ccc'])
#
# table['a':'c', 'aa':'bb'] = np.ones(shape=(2, 1, 3))
#
# table.save(path=path)
#
# table2 = Table.load(path)
#
# print(table)
# print(table2)

# table['a', 'bb', 'ccc'] = 1
# table['a':'c', 'aa':'bb'] = np.ones(shape=(2, 1, 3))
# table['a', 'bb', :] = [object()] * 3
#
# x = table['a']
# y = table['a':'c', 'aa':'bb']
# z = table[:, :, :]
#
# print(x)
# print(y)
# print(z)

# print(table)
