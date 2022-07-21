import torch
import torch.nn as nn
class RevSequential(nn.ModuleList):
    '''
    功能大部分与ModuleList重叠
    '''
    def __init__(self, modules=None):
        super().__init__(modules)

    def append(self, module):
        assert hasattr(module, 'invert') and callable(module.invert)
        super().append(module)

    def extend(self, modules):
        for m in modules:
            self.append(m)

    def forward(self, x1, x2):
        y1, y2 = x1, x2
        for m in self:
            y1, y2 = m(y1, y2)
        return y1, y2

    def invert(self, y1, y2):
        x1, x2 = y1, y2
        for m in self[::-1]:
            x1, x2 = m.invert(x1, x2)
        return x1, x2

class RevGroupBlock(nn.ModuleList):
    '''
    当前只支持输入通道等于输出通道，并且不允许下采样
    '''
    def __init__(self,in_channel,out_channel,stride,act,block_type,blocks,**kwargs):
        assert in_channel == out_channel    # 输入通道必须和输出通道一样
        assert stride == 1
        self.mods = []
        for _ in range(blocks):
            self.mods.append(block_type(in_channel=in_channel,out_channel=out_channel,stride=1,act=act,**kwargs))
        super().__init__(self.mods)
    def forward(self,x):
        for m in self.mods:
            x = m(x)
        return x


