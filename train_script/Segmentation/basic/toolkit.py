"""
本文件都是一些无实际意义的、仅为了简化代码的基础工具
"""


class Dict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__()
        for arg in [a for a in args if hasattr(a, 'keys')]:
            super().update(arg)
        super().update([a for a in args if not hasattr(a, 'keys')])
        super().update(kwargs)

    def __getattr__(self, arg_name):
        return self[arg_name]

    def __getitem__(self, item):
        return super().__getitem__(item) if item in self else None


class InterfaceProxy(object):
    def __init__(self, **kwargs):
        self.__setattr__('__call__', self.NOT_IMPLEMENT)
        self.__setattr__('__add__', self.NOT_IMPLEMENT)
        for name, impl in kwargs.items():
            self.__setattr__(name, impl)

    """
    此处出现了某种脏语法,该现象的实质在于, call 和 add 这种函数的调用
    是在 class 级别进行的,也就是说,它们总是调用进入写在函数里面的方法
    然而,由于属性 Has been rewrited, when call it again
    it may use __getattribute__ and find the property instead
    as known, properties always implemented
    """

    def __call__(self, *args, **kwargs):
        return self.__call__(*args, **kwargs)

    def __add__(self, *args, **kwargs):
        return self.__add__(*args, **kwargs)

    @staticmethod
    def NOT_IMPLEMENT(*args, **kwargs):
        raise NotImplementedError('The method has not implemented!')

    @staticmethod
    def EMPTY(*args, **kwargs):
        pass

    @staticmethod
    def IDENTITY(*args, **kwargs):
        if len(args) == 0:
            return kwargs or None
        elif len(args) == 1:
            return args[0]
        else:
            return args

    @staticmethod
    def ZERO(*args, **kwargs):
        return 0
