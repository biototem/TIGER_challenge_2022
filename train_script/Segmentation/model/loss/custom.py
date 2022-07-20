import numpy as np
from typing import Tuple

import torch
from torch import nn

from basic import config
from .cross_entropy import CrossEntropy
from .dice import Dice, GenDice
from .standard import StandardDice


class Combine(nn.Module):
    """
    组合Loss
    """
    def __init__(self, alpha: float = 1, beta: float = 1, gamma: float = 1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ce = CrossEntropy()
        self.dc = Dice()
        self.gd = GenDice()
        self.__name__ = 'combine-%.2f-%.2f-%.2f' % (alpha, beta, gamma)

    def forward(self, pr, gt, *args):
        ce = self.ce(pr, gt, *args)
        dc = self.dc(pr, gt, *args)
        gd = self.gd(pr, gt, *args)
        return ce * self.alpha + dc * self.beta + gd * self.gamma


class LCEDice(nn.Module):
    """
    组合Loss
    """
    def __init__(self, alpha: float = 2):
        super().__init__()
        self.alpha = alpha
        smooth = config['model.label_smooth'] or 0.04
        limit = config['model.ce_limit'] or 5
        weights = config['model.label_weights'] or None
        # self.ce = LabelSmoothLimitedCrossEntropy(smooth=0.04, weights=(6, 60, 10, 30))
        self.ce = LabelSmoothLimitedCrossEntropy(smooth=smooth, limit=limit, weights=weights)
        # self.dc = Dice()
        self.dc = StandardDice(ignore=0)
        self.__name__ = 'limit-ce-dice-%.2f' % alpha

    def forward(self, pr, gt, *args):
        # 说明： 虽然 ce 的理论计算值可以大于 1
        # 但我们设置了 limit = 5，也就是说，它的计算值无论如何不会大于 5
        # For dropping label == 0, here new gr instead of gt
        gt[:, 0, :, :] = 0
        ce = self.ce(pr, gt, *args)
        dc = self.dc(pr, gt, *args)
        return ce * self.alpha + dc


class LabelSmoothLimitedCrossEntropy(nn.Module):
    """
    标签平滑最大值抑制交叉熵
    关于 smooth、 limit、 weights 的具体表现，可以在 test/损失函数测试.py 中观察
    极值点并不总是出现在 1-smooth 处，其位置由 smooth 决定、受 limit、 weights 影响
    提高 limit 会增加 0、1 附近的损失值
    权重高的类别，极值点将向 1 偏移； 反之则向 0 偏移
    但如果 smooth == 0，这些参数将只影响预测曲线的陡峭程度和值域
    """
    def __init__(self, smooth: float = 0.04, limit: float = 5, weights: Tuple[float, ...] = None):
        super().__init__()
        self.classes = config['dataset.class_num']
        # 令 1-s1 = 1-s2+s2/c
        # 解得 s2 = s1 * c/(c-1)
        # 在 forward 中使用 s1 不方便，因此用 s2 代替
        assert self.classes > 1, '使用本损失函数时，类型数必须大于1'
        self.smooth = smooth * self.classes / (self.classes - 1)
        self.limit = torch.e ** -limit
        self.weights = weights or (1,) * self.classes
        self.weights = tuple(w / np.mean(self.weights) for w in self.weights)
        self.__name__ = 'smooth-limited-ce'

    def forward(self, pr: torch.tensor, gt: torch.tensor, *args):
        mask = gt.type(torch.int32).sum(dim=1, keepdim=True)
        # 将 pr 限定在有梯度的区域
        gt = (1 - self.smooth) * gt + self.smooth / self.classes
        gt = gt * mask
        loss = gt * torch.log(pr * (1 - self.limit) + self.limit)
        for i, w in enumerate(self.weights):
            loss[:, i, :, :] *= w
        loss = -loss.sum()
        # 对 置信度 进行 归一
        loss = loss / (mask.sum() + 1e-17)
        return loss
