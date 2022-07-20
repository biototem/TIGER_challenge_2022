from segmentation_models_pytorch.utils import base
from basic import config


class MFscore(base.Metric):
    """
    Fsocre = (1+bb) * pre * rec / (bb*pre + rec)
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
        self.only = only if only else tuple(range(config['dataset.class_num']))
        self._name = name if name else f'mf({beta})score-{only}'

    def forward(self, pr, gt, *args):
        pr = pr[:, self.only, :, :]
        gt = gt[:, self.only, :, :]
        tp = pr * gt + 1e-17
        score = (1+self.bb) * tp.sum((0, 2, 3)) / (pr.sum((0, 2, 3)) + self.bb * gt.sum((0, 2, 3)))
        return score.mean()
