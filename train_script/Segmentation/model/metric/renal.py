import torch

from basic import config
from segmentation_models_pytorch.utils import base

from utils import one_hot_torch


class Renal(base.Metric):
    """
    本评价函数专用于统计肾小管预测的 Dice 指标
    """
    def __init__(self):
        assert config['dataset.class_num'] == 4, '业务可能已变更，此处代码应当伴随修改'
        super().__init__()
        self.classes = config['dataset.class_num']
        self._name = 'dice-on-renal'

    def forward(self, pr: torch.Tensor, gt: torch.Tensor, *args):
        # 首先计算蒙版
        mask = gt.sum(dim=(1,), keepdim=True)
        # 然后合并细胞
        gt = gt[:, (2, 3), :, :].sum(dim=(1,), keepdim=True)
        tmp = torch.zeros_like(pr)
        tmp[:, (0, 1), :, :] = pr[:, (0, 1), :, :]
        tmp[:, 2, :, :] = pr[:, (2, 3), :, :].sum(dim=1)
        # 然后 argmax 离散
        pr = one_hot_torch(torch.argmax(tmp, dim=1), self.classes).permute(0, 3, 1, 2).cuda()
        # 然后选取层
        pr = pr[:, (2,), :, :]
        i = pr * gt
        u = mask*(pr+gt)/2 + 1e-17
        dice = i.sum((0, 2, 3)) / u.sum((0, 2, 3))
        return dice.mean()
