import torch
import numpy as np
from basic import config
from segmentation_models_pytorch.utils import base


class Special(base.Metric):
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

    def __init__(self, tag: str):
        super().__init__()
        assert tag in ['score', 'dice_tumor', 'dice_stroma'], '仅支持特定标签'
        self.classes = config['dataset.class_num']
        self.tag = tag
        self._name = f'competition_{tag}'

    def forward(self, pr: torch.Tensor, gt: torch.Tensor, *args):
        # 评测规则如下：
        # dice1: pr & gt == 1
        # dice2: pr & gt == 2、6
        # dice: mean(dice1, dice2)
        mask = 1 - gt[:, 0, :, :]
        pr = torch.argmax(pr, dim=1)
        pr = torch.nn.functional.one_hot(pr, num_classes=gt.shape[1])
        pr = torch.permute(pr, (0, 3, 1, 2))
        dice_tumor = self.dice(pr[:, 1, :, :], gt[:, 1, :, :], mask)
        dice_stroma = self.dice(pr[:, 2, :, :] | pr[:, 6, :, :], gt[:, 2, :, :] | gt[:, 6, :, :], mask)
        dice = (dice_tumor + dice_stroma) / 2
        if self.tag == 'score':
            return dice
        if self.tag == 'dice_tumor':
            return dice_tumor
        if self.tag == 'dice_stroma':
            return dice_stroma
        raise ValueError

    @staticmethod
    def dice(pr, gt, mask):
        # pr = pr > 0.5
        inter = mask * pr * gt
        union = mask * (pr + gt) / 2 + 1e-17
        return inter.sum() / union.sum()

    def calculate(self, pr: np.ndarray, gt: np.ndarray, *args):
        mask = 1 - gt[:, 0, :, :]
        dice1 = self.dice(pr[:, 1, :, :], gt[:, 1, :, :], mask)
        dice2 = self.dice(pr[:, 2, :, :] + pr[:, 6, :, :], gt[:, 2, :, :] + gt[:, 6, :, :], mask)
        dice = (dice1 + dice1) / 2
        return [round(float(x), 3) for x in (dice, dice1, dice2)]
