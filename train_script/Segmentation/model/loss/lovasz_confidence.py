import torch
from torch.autograd import Variable

from basic import config

try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse as ifilterfalse
from torch.nn.modules.loss import _Loss


class Lovasz(_Loss):
    def __init__(self):
        super().__init__()
        self.classes = config['dataset.class_num']
        self.__name__ = 'lovasz-confidence'

    def forward(self, pr, gt, *args):
        # 洛瓦兹的蒙版方法，只需gt和pr同时置0即可
        mask = gt.sum(dim=(1,), keepdim=True)
        pr = pr * mask
        gt = gt.type(torch.int32)

        losses = []
        pr = pr.permute(1, 0, 2, 3).contiguous().view(self.classes, -1)
        gt = gt.permute(1, 0, 2, 3).contiguous().view(self.classes, -1)
        for c in range(self.classes):
            if gt[c, :].sum() == 0:
                continue
            error = (Variable(gt[c, :]) - pr[c, :]).abs()
            sort_error,  sort_index = torch.sort(error, 0, descending=True)
            sort_index = sort_index.data
            sort_gt = gt[c, sort_index]
            losses.append(torch.dot(sort_error, Variable(lovasz_grad(sort_gt))))
        if len(losses) == 0:
            return 0
        return sum(losses) / len(losses)


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard
