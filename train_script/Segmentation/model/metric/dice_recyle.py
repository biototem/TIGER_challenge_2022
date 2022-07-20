from basic import config
from segmentation_models_pytorch.utils import base


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
        self.only = only if only else tuple(range(config['dataset.class_num']))
        self._name = name if name else 'dice-{}'.format(only)

    def forward(self, pr, gt, *args):
        pr = pr[:, self.only, :, :]
        gt = gt[:, self.only, :, :]
        i = pr * gt + 1e-17
        u = (pr+gt)/2 + 1e-17
        dice = i.sum((0, 2, 3)) / u.sum((0, 2, 3))
        return dice.mean()
