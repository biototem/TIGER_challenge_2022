import torch
import numpy as np
from segmentation_models_pytorch.utils import base
from basic import config
from utils import one_hot_torch, one_hot_numpy


class MFscore(base.Metric):
    """
    Fscore = (1+bb) * pre * rec / (bb*pre + rec)
    when b!=1:
     => Fsocre = (1+bb) * TP/(TP+FP) * TP/(TP+FN) / (bb*TP/(TP+FP) + TP/(TP+FN))
     => Fsocre = (1+bb) * TP / [bb*(TP+FN) + (TP+FP)]
     => Fsocre = TP / [TP + (FP + bb*FN)/(1+bb)]
     => Fsocre = TP / [TP + (pr-TP + bb*(gt-TP))/(1+bb)]
     => Fsocre = (1+bb) * TP / (pr + bb*gt)
    """

    def __init__(self, beta: float = 1, only: tuple = None, name: str = ''):
        super().__init__()
        self.bb = beta * beta
        self.classes = config['dataset.class_num']
        self.only = only or tuple(range(self.classes))
        self._name = name or f'mf({beta})score-{only}'

    def forward(self, pr: torch.Tensor, gt: torch.Tensor, *args):
        pr = one_hot_torch(torch.argmax(pr, dim=1), self.classes).permute(0, 3, 1, 2).cuda()
        mask = gt.sum(dim=(1,), keepdim=True)
        pr = pr[:, self.only, :, :] * mask
        gt = gt[:, self.only, :, :] * mask
        tp = pr * gt
        pr = pr.sum((0, 2, 3))
        gt = gt.sum((0, 2, 3))
        tp = tp.sum((0, 2, 3))

        score = (1+self.bb) * tp / (pr + self.bb * gt + 1e-17)
        return score.mean()

    def calculate(self, pr: np.ndarray, gt: np.ndarray, *args):
        # pr = one_hot_numpy(np.argmax(pr, axis=1), self.classes).transpose((0, 3, 1, 2))
        # mask = gt.sum(axis=(1,), keepdims=True)
        pr = pr[:, self.only, :, :]
        gt = gt[:, self.only, :, :]
        tp = pr * gt
        pr = pr.sum((0, 2, 3))
        gt = gt.sum((0, 2, 3))
        tp = tp.sum((0, 2, 3))
        score = (1+self.bb) * tp / (pr + self.bb * gt + 1e-17)
        del pr
        del gt
        del tp
        return score.mean()
