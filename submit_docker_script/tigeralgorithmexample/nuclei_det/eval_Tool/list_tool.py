from typing import Iterable
import math


def list_multi_get_with_ids(self: list, ids: Iterable):
    return [self[i] for i in ids]


def list_multi_get_with_bool(self: list, bools: Iterable):
    assert len(self) == len(bools)
    a = [self[i] for i, b in enumerate(bools) if b]
    return a


def list_multi_set_with_ids(self: list, ids: Iterable, items: Iterable):
    assert len(ids) == len(items)
    for _id, item in zip(ids, items):
        self[_id] = item


def list_multi_set_with_bool(self: list, bools: Iterable, items: Iterable):
    assert len(self) == len(bools)
    wait_set_ids = []
    for i, b in enumerate(bools):
        if b:
            wait_set_ids.append(i)
    assert len(wait_set_ids) == len(items)
    for i, item in zip(wait_set_ids, items):
        self[i] = item


def list_bool_to_ids(self: Iterable):
    ids = []
    for i, it in enumerate(self):
        if it == True:
            ids.append(i)
    return ids


def int_list(self: Iterable):
    return [int(i) for i in self]


def float_list(self: Iterable):
    return [float(i) for i in self]


def list_split_by_size(self: Iterable, size: int):
    self = list(self)
    g = []
    i = 0
    while True:
        s = self[i*size: (i+1)*size]
        i+=1
        if len(s) == size:
            g.append(s)
        elif len(s) > 0:
            g.append(s)
            break
        else:
            break
    return g


def list_split_by_group(self: Iterable, n_group: int):
    self = list(self)
    sizes = [int(len(self) / n_group)] * n_group
    for i in range(len(self) % n_group):
        sizes[i] += 1
    g = []
    i = 0
    for s in sizes:
        l = self[i: i+s]
        g.append(l)
        i += s
    return g


if __name__ == '__main__':
    a = [1, 2, 3, 4, 5, 6]
    b = list_multi_get_with_ids(a, [0, 2, 4])
    assert a[0] == b[0] and a[2] == b[1] and a[4] == b[2]

    c = list_multi_get_with_bool(a, [True, False, True, False, False, False])
    assert a[0] == c[0] and a[2] == c[1]

    list_multi_set_with_ids(a, [0, 2], [3, 1])
    assert a[0] == 3 and a[2] == 1

    list_multi_set_with_bool(a, [True, False, True, False, False, False], [1, 3])
    assert a[0] == 1 and a[2] == 3

    d = list_bool_to_ids([True, False, True, False, False, False])
    assert len(d) == 2 and d[0] == 0 and d[1] == 2

    b = list_split_by_size(a, 4)
    assert b == [[1, 2, 3, 4], [5, 6]]

    b = list_split_by_group(a, 2)
    assert b == [[1, 2, 3], [4, 5, 6]]
