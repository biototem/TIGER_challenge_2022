import torch
import numpy as np
from basic import config
from segmentation_models_pytorch.utils import base

from utils import one_hot_torch, one_hot_numpy


class MDice(base.Metric):
    """
    Fsocre = (1+bb) * pre * rec / (bb*pre + rec)
    when b=1:
     => Fsocre = 2 * (pre * rec) / (pre + rec)
     => Fsocre = 2 / (1/pre + 1/rec)
     => Fsocre = 2 / [(TP+FP)/TP + (TP+FN)/TP]
     => Fsocre = 2 * TP / (FP + FN + 2*TP)
     => Fsocre = Dice
    so Dice is F1-score
    means MDice is M-F1-score
    """

    def __init__(self, only: tuple = None, name: str = ''):
        super().__init__()
        self.classes = config['dataset.class_num']
        self.only = only if only else tuple(range(self.classes))
        self._name = name if name else 'dice-{}'.format(only)

    def forward(self, pr: torch.Tensor, gt: torch.Tensor, *args):
        pr = one_hot_torch(torch.argmax(pr, dim=1), self.classes).permute(0, 3, 1, 2).cuda()
        gt = one_hot_torch(gt, self.classes).permute(0, 3, 1, 2).cuda()
        # mask = gt.sum(dim=(1,), keepdim=True)
        mask = ~gt[:, 0, :, :]
        pr = pr[:, self.only, :, :]
        gt = gt[:, self.only, :, :]
        i = pr * gt
        u = mask*(pr+gt)/2 + 1e-17
        dice = i.sum((0, 2, 3)) / u.sum((0, 2, 3))
        return dice.mean()

    def calculate(self, pr: np.ndarray, gt: np.ndarray, *args):
        # pr = one_hot_numpy(np.argmax(pr, axis=1), self.classes).transpose((0, 3, 1, 2))
        # mask = gt.sum(axis=(1,), keepdims=True)
        pr = pr[:, self.only, :, :]
        gt = gt[:, self.only, :, :]
        i = pr * gt
        u = (pr+gt)/2 + 1e-17
        dice = i.sum((0, 2, 3)) / u.sum((0, 2, 3))
        del pr
        del gt
        del i
        del u
        return dice.mean()
