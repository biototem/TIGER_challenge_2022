from basic import config
import numpy as np
import torch
from torch import nn


class Dice(nn.Module):
    def __init__(self, weights: tuple = None):
        super().__init__()
        if not weights:
            weights = (1,) * config['dataset.class_num']
        self.weights = np.array(weights, dtype=np.float32)
        self.weights /= self.weights.mean()
        self.weights = torch.tensor(self.weights).cuda()
        self.__name__ = 'dice-loss'

    def forward(self, pr, gt, *args):
        mask = gt.sum(dim=(1,), keepdim=True)
        i = pr * gt + 1e-17
        u = (pr * mask + gt)/2 + 1e-17
        dice = i.sum((0, 2, 3)) / u.sum((0, 2, 3))
        dice = dice * self.weights
        return 1 - dice.mean()


class GenDice(nn.Module):
    def __init__(self):
        super().__init__()
        self.__name__ = 'gen-dice-loss'

    @staticmethod
    def forward(pr, gt, *args):
        mask = gt.sum(dim=(1,), keepdim=True)
        i = pr * gt + 1e-17
        u = (pr * mask + gt)/2 + 1e-17
        dice = i.sum((0, 2, 3)) / u.sum((0, 2, 3))

        weight = gt.type(torch.int32).sum((0, 2, 3))
        weight = 1 / (1 + weight**2)
        weight = weight / weight.sum()

        dice = dice * weight
        return 1 - dice.sum()


class ____________GenWrongDice(nn.Module):
    def __init__(self):
        super().__init__()
        self.__name__ = 'gen-wrong-dice-loss'

    @staticmethod
    def forward(pr, gt, *args):
        mask = gt.sum(dim=(1,), keepdim=True)
        weight = gt.sum((0, 2, 3))
        weight = weight / weight.sum()
        i = pr * gt + 1e-17
        u = mask*(pr+gt)/2 + 1e-17
        dice = i.sum((0, 2, 3)) / u.sum((0, 2, 3))
        dice = dice * weight
        return 1 - dice.sum()


# 我敢觉这个没用,弃置掉算了
# https: // blog.csdn.net / JMU_Ma / article / details / 97533768
class SoftDice(nn.Module):
    def __init__(self):
        super().__init__()
        self.__name__ = 'soft-dice-loss'

    @staticmethod
    def forward(pr, gt, *args):
        num = gt.size(0)
        pr = pr.view(num, -1)
        gt = gt.view(num, -1)
        score = (2. * (pr * gt).sum(1) + 1e-17) / (pr.sum(1) + gt.sum(1) + 1e-17)
        score = 1 - score.sum() / num
        return score
