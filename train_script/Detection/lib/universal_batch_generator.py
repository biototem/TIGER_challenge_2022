'''
用于把生成器的多个输出打包成多个批量
例如
生成器1生成 (1,2,3) (4,5,6)
universal_batch_generator 将会输出 (1,4) (2,5) (3,6)
'''


def universal_batch_generator(g, batch_size):
    elems = next(g)
    assert isinstance(elems, tuple), '请务必确保生成器返回值为输出元组'
    batch_bulk = [[i] for i in elems]

    for items in g:
        assert len(items) == len(batch_bulk)
        if len(batch_bulk[0]) == batch_size:
            # 注意线程安全，所以需要返回 batch_bulk 的浅复制副本
            yield tuple(batch_bulk)
            for i in range(len(batch_bulk)):
                batch_bulk[i] = []

        for i, bulk in zip(items, batch_bulk):
            bulk.append(i)

    if len(batch_bulk[0]) > 0:
        yield tuple(batch_bulk)
        for i in range(len(batch_bulk)):
            batch_bulk[i] = []


if __name__ == '__main__':
    def gen1(n_out, epochs):
        '''
        测试用任意生成器
        :param n_out: 多少个输出
        :param epochs: 多少轮
        :return:
        '''
        for e in range(epochs):
            out = (e,) * n_out
            yield out

    g = gen1(5, 100)
    g2 = universal_batch_generator(g, 6)
    for i in g2:
        print(i)
