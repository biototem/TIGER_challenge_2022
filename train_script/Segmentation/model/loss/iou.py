from torch import nn


# 源码及讲解参考 - license[CC BY-SA 4.0]
# https://blog.csdn.net/CaiDaoqing/article/details/90457197
class SoftIoU(nn.Module):
    def __init__(self, n_classes):
        super(SoftIoU, self).__init__()
        self.n_classes = n_classes
        self.__name__ = 'cross-iou-loss'

    def forward(self, pr, gt, *args):
        mask = gt.sum(dim=(1,), keepdim=True)
        # Numerator Product
        inter = pr * gt
        # Sum over all pixels N x C x H x W => N x C
        inter = inter.view(gt.size(0), self.n_classes, -1).sum(2)
        # Denominator
        union = pr + gt - (pr * gt)
        union = union * mask
        # Sum over all pixels N x C x H x W => N x C
        union = union.view(gt.size(0), self.n_classes, -1).sum(2)
        loss = inter / (union + 1e-16)
        # Return average loss over classes and batch
        return -loss.mean()
