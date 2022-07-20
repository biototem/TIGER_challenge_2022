from typing import TypeVar
import numpy as np
import abc
import sys
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from basic import config
from utils import one_hot_numpy


M = TypeVar('M', bound=torch.nn.Module)


class TrainEpoch(object):
    """
    用于执行训练过程
    @param train_batch：数据集的 batch_size 受限于训练时显存，无法随意扩展，因此使用 train_batch 变相扩增数据集
    """
    def __init__(self, model: M, loss: M, optimizer: M, scheduler: M):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.verbose = config['train.tqdm_verbose']

        # 找设备
        self.device = torch.device(config['train.device'])
        self.model.to(self.device)
        self.loss.to(self.device)

    def get_lr(self) -> float:
        return tuple(map(lambda p: p['lr'], self.optimizer.param_groups))

    def run(self, dataloader: DataLoader, epoch: int) -> None:
        # 每代开始前调整学习率
        self.scheduler.step(epoch)
        # 确保模型处在训练状态
        self.model.train()
        # 计算平均 loss
        sum_loss = 0.
        patch_count = 0
        with tqdm(dataloader, desc=f'training-{epoch + 1}', file=sys.stdout, disable=not self.verbose) as iterator:
            for images, labels in iterator:
                self.optimizer.zero_grad()
                # 确保设备一致
                images, labels = images.to(self.device), labels.to(self.device)
                patch_size = images.shape[0]

                # 开始预测，注意，模型可能有辅助网络，也可能没有，因此返回值具有任意性
                # TODO: 可以优化辅助网络的显存占用：只需先 loss1.backward()，再 loss2.backward() 即可
                # TODO： 但是我不知道该如何设计这样的代码 -> 这会导致 train.py 强依赖于 net.py，或者相反，这不符合模块化设计理念
                all_predicts = self.model(images)

                # 统一用 tuple 处理
                all_predicts = all_predicts if isinstance(all_predicts, (tuple, list)) else (all_predicts,)

                # 执行更新
                # 无论返回几个结果，梯度都是相加关系
                # 不过在运行时日志中可能会比较关心正常网络和辅助网络谁的 loss 更小这个问题
                losses = []
                loss = 0
                for predicts in all_predicts:
                    p_loss = self.loss(predicts, labels)
                    loss = loss + p_loss
                    losses.append(p_loss.item())
                loss.backward()
                loss = loss.item()
                self.optimizer.step()

                del images, labels, all_predicts, predicts, p_loss

                # 计算平均 loss
                patch_count += patch_size
                sum_loss += loss
                if self.verbose:
                    mean_loss = '%.2f' % (sum_loss / patch_count)
                    loss = '%.2f' % (loss / patch_size)
                    losses = ['%.2f' % (l / patch_size) for l in losses]
                    iterator.set_postfix_str(
                        f'epoch: {epoch}, '
                        f'loss: {mean_loss}, '
                        f'batch-loss: {loss}, '
                        f'divide-loss: {losses}'
                    )
