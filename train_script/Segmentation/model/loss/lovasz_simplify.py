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
        """
        原lovasz softmax需要提供的标签为索引值（n,h,w），这里魔改成提供onehot编码，方便计算IoU等指标
        y_pred:(n,c,h,w)，应为softmax(dim=1)的值
        y_true:(n,c,h,w)，应为onehot编码
        """
        super().__init__()
        self.classes = config['dataset.class_num']
        self.__name__ = 'lovasz-loss'

    def forward(self, pr, gt, *args):
        """
         Multi-class Lovasz-Softmax loss
           probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
           labels: [P] Tensor, ground truth labels (between 0 and C - 1)
           classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
         """
        losses = []
        pr = pr.permute(1, 0, 2, 3).contiguous().view(self.classes, -1)
        gt = gt.permute(1, 0, 2, 3).contiguous().view(self.classes, -1)
        for c in range(self.claases):
            if not torch.any(gt[c, :].type(torch.bool)):
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
