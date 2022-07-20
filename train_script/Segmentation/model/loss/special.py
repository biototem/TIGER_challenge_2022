import torch
import numpy as np
from torch import nn
import math

from basic import config


class Special(nn.Module):
    """
    比赛规则中，分割模型只评价 侵蚀性肿瘤(1)、肿瘤相关间质(2)、炎症基质(6)，其余不予评价
    且肿瘤相关间质和炎症基质满足：
    “For the evaluation there will be no differentiation between both stroma types - they will be grouped together”
    针对此规则，特别的编写一个 loss 函数进行训练
    针对1: 重点区分侵蚀性肿瘤、非侵蚀性肿瘤
    针对2: 不区分两种间质
    """

    def __init__(self, weights=None):
        super().__init__()
        if not weights:
            weights = config['model.label_weights'] or (1,) * config['dataset.class_num']
        self.weights = np.array(weights, dtype=np.float32)
        self.weights = torch.tensor(self.weights).cuda()
        self.__name__ = 'competition_loss'
        self.limit = torch.e ** -15
        n = 11
        self.kernel = [math.comb(n, i) for i in range(n)]
        self.kernel = torch.tensor([self.kernel], dtype=torch.float32, requires_grad=False)
        self.kernel = torch.matmul(self.kernel.T, self.kernel)
        self.kernel /= self.kernel.max()
        self.kernel = self.kernel.cuda()
        # self.kernel = torch.unsqueeze(self.kernel, dim=0)

    def forward(self, pr, gt, *args):
        # 标签有以下可能：
        # gt == 0 不参与评价
        # gt == 1 浸润性肿瘤
        # gt == 2、6 不区分的间质
        # gt == 3、4、5、7 负样本
        # 评测损失值由四部分组成：
        # gt == 1 && pr != 1
        # gt != 1 && pr == 1
        # gt == 2、6 && pr != 2、6
        # gt != 2、6 && pr == 2、6
        # 这四部分的权重是均等的
        # 所以最简单的、将它直接拆解为两个二分类 ce
        # 再加上无足轻重的全局多分类 ce 辅助分类
        mask = 1 - gt[:, 0, :, :]
        gravitation = torch.empty_like(pr, requires_grad=False)
        class_num = pr.shape[1]
        kernel = self.kernel.expand([pr.shape[1], pr.shape[1], -1, -1])
        gravitation = torch.conv2d(
            input=gt.type(torch.float32),
            weight=kernel,
            padding="same",
        )
        ce1 = self.bce(pr[:, 1, :, :], gt[:, 1, :, :], mask, gravitation[:, 1, :, :])
        ce2 = self.bce(
            pr[:, 2, :, :] + pr[:, 6, :, :],
            gt[:, 2, :, :] + gt[:, 6, :, :],
            mask,
            gravitation[:, 2, :, :] + gravitation[:, 6, :, :],
        )
        ce3 = self.mce(pr, gt, mask, gravitation)
        # 对置信度归一
        return ce1 + ce2 + 0.2 * ce3

    def bce(self, pr, gt, mask, gravitation):
        loss_pos = mask * gt * torch.log(self.limit + pr)
        loss_neg = mask * (1 - gt) * torch.log(self.limit + 1 - pr)
        loss = loss_pos + loss_neg
        loss *= gravitation
        return -loss.mean()

    def mce(self, pr, gt, mask, gravitation):
        mask = torch.unsqueeze(mask, 1)
        loss = mask * gt * torch.log(self.limit + pr)
        loss *= gravitation
        return -loss.mean()
