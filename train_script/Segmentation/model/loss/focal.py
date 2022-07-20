import torch
from torch import nn


# 我感觉这个好像就是tensorflow默认的那个loss,但还没改
# https://blog.csdn.net/CaiDaoqing/article/details/90457197
class Focal(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_fn = nn.CrossEntropyLoss(weight=weight)
        self.__name__ = 'focal-loss'

    def forward(self, pr, gt, *args):
        log_pt = -self.ce_fn(pr, gt)
        pt = torch.exp(log_pt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * log_pt
        return loss
