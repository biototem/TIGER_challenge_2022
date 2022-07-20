import torch
import numpy as np
from torch import nn

from basic import config


from torch.nn import functional as F


# class CrossEntropy(torch.nn.CrossEntropyLoss):
#     """
#     原nn.CrossEntropyLoss需要提供的标签为索引值（n,h,w），这里魔改成提供onehot编码，方便计算IoU等指标
#     input:(n,c,h,w)，应为softmax(dim=1)的值
#     target:(n,c,h,w)，应为onehot编码
#     weight:(c),If given, has to be a Tensor of size `c`
#     """
#
#     def __init__(self, weight=None, size_average=None, ignore_index: int = -100,
#                  reduce=None, reduction: str = 'mean'):
#         # UserWarning: size_average and reduce args will be deprecated, please use reduction='mean' instead.
#         #   warnings.warn(warning.format(ret))
#         super(CrossEntropy, self).__init__(weight, size_average, reduce, reduction)
#         self.ignore_index = ignore_index
#         self.__name__ = 'cross-entropy-loss'
#
#     def forward(self, inputs, target, *args):
#         target = torch.argmax(target, dim=1)
#         # 好费解，这里的input是one_hot编码，但target不是
#         return F.cross_entropy(
#             input=inputs,
#             target=target,
#             weight=self.weight,
#             ignore_index=self.ignore_index,
#             reduction=self.reduction
#         )


class CrossEntropy(nn.Module):
    """
    原nn.CrossEntropyLoss需要提供的标签为索引值（n,h,w），这里魔改成提供onehot编码，方便计算IoU等指标
    input:(n,c,h,w)，应为softmax(dim=1)的值
    target:(n,c,h,w)，应为onehot编码
    weight:(c),If given, has to be a Tensor of size `c`
    """

    def __init__(self, weights=None):
        super().__init__()
        if not weights:
            weights = (1,) * config['dataset.class_num']
        self.weights = np.array(weights, dtype=np.float32)
        self.weights = torch.tensor(self.weights).cuda()
        self.__name__ = 'cross-entropy-loss'

    def forward(self, pr, gt, *args):
        # 将 pr 限定在有梯度的区域
        # pr = 1e-17 + (pr * (1-1e-9))
        loss = gt * torch.log(pr + 1e-9)
        loss = loss.sum((0, 2, 3)) * self.weights
        loss = -loss.sum()
        # 对 置信度 进行 归一
        loss = loss / (gt.type(torch.int32).sum() + 1e-17)
        return loss
