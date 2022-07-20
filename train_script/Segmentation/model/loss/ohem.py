目前为止,这个损失函数还没有测试过



from basic import config
import numpy as np
import torch

# OHEM 的方法是 只训练困难样本
# # 源码及讲解参考 - license[CC BY-SA 4.0]
# # https://blog.csdn.net/CaiDaoqing/article/details/90457197
# class OHEMLoss(nn.Module):
#     def __init__(self, alpha=0.5, gamma=2, weight=None, ignore_index=255, percent=0.5):
#         super().__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.weight = weight
#         self.ignore_index = ignore_index
#         self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)
#         self.__name__ = 'OHEM-loss'
#
#     def forward(self, preds, labels):
#         preds = preds.contiguous().view(-1)
#         labels = labels.contiguous().view(-1)
#
#         max_val = (-preds).clamp(min=0)
#         loss = preds - preds * labels + max_val + ((-max_val).exp() + (-preds - max_val).exp()).log()
#
#         # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
#         invprobs = F.logsigmoid(-preds * (labels * 2 - 1))
#         focal_loss = self.alpha * (invprobs * self.gamma).exp() * loss
#
#         # Online Hard Example Mining: top x% losses (pixel-wise). Refer to http://www.robots.ox.ac.uk/~tvg/publications/2017/0026.pdf
#         OHEM, _ = focal_loss.topk(k=int(self.percent * [*focal_loss.shape][0]))
#         return OHEM.mean()
