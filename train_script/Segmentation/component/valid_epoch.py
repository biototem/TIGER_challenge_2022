import numpy as np
import abc
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from basic import hyp
from model import optimizer, scheduler
from predict import PredictManager
from predict.merger import TorchMerger
from utils import one_hot_numpy


class MyEpoch(object):
    """
    用于执行训练过程
    """
    def __init__(self, model, loss, metrics: tuple = ()):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = 'epoch'
        self.verbose = hyp['train.tqdm_verbose']
        self.device = torch.device(hyp['train.device'])

        # 全部启用 gpu
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

        # 现在，用高斯融合后的结果作为模型的验证选择标准
        self.merger = TorchMerger()

    @abc.abstractmethod
    def batch_update(self, image, label):
        raise NotImplementedError

    def on_epoch_start(self, epoch):
        self.model.eval()

    def on_epoch_end(self, logs):
        pass

    def __predict__(self, loader):
        for i, (grids, patches) in enumerate(loader):
            results = self.model(patches.cuda())
            self.merger.set(results, grids)

    def run(self, dataloader, epoch):
        self.on_epoch_start(epoch)
        logs = {self.loss.__name__: 0}
        with torch.no_grad():
            with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not self.verbose) as iterator:
                labels = []
                predicts = []
                validates = {}
                patch_count = 0
                for subset in iterator:
                    loader = DataLoader(
                        subset,
                        batch_size=hyp['train.batch_size'],
                        num_workers=hyp['train.num_workers']
                    )
                    h, w = subset.get_HW()
                    with self.merger.with_shape(W=w, H=h):
                        self.__predict__(loader)
                        predict = self.merger.tail()
                    label = subset.get_label()
                    # 只选取 box 区域用于验证
                    l, u, r, d = subset.get_box()
                    predict = predict[u: d, l: r, :]
                    label = label[u: d, l: r, :]
                    # 评估损失函数
                    loss = self.loss.caculate(predict, label)
                    # 基于图块平均
                    logs[self.loss.__name__] += loss.item() * ()
                # 按图块评估评价函数
                label = label.cpu().detach().numpy()
                predict = predict.cpu().detach().numpy()
                predict = one_hot_numpy(np.argmax(predict, axis=1), self.classes).transpose((0, 3, 1, 2))
                for i, name in enumerate(names):
                    # print(name)
                    if bool(labels):
                        # 进来新玩意了， 终结老朋友， 如果老朋友存在的话
                        stack_label = np.stack(labels, axis=0)
                        stack_predict = np.stack(predicts, axis=0)
                        predicts.clear()
                        labels.clear()
                        validates[name] = {
                            'lab_pixes': stack_label.sum(axis=(0, 2, 3)),
                            'pre_pixes': stack_predict.sum(axis=(0, 2, 3)),
                        }
                        for metric in self.metrics:
                            validates[name][metric.__name__] = metric.calculate(stack_predict, stack_label).item()
                        del stack_label
                        del stack_predict
                    # 进来老朋友了，先暂存起来
                    labels.append(label[i, :, :, :])
                    predicts.append(predict[i, :, :, :])
                    continue
                # 实时统计
                for metric in self.metrics:
                    # 只有 only 类型中囊括 lab 的， 才列入 img 评估
                    evaluate_img = [
                        v[metric.__name__] for v in validates.values()
                        if v['lab_pixes'][(metric.only, )].sum() > 0
                    ]
                    logs[metric.__name__] = sum(evaluate_img) / (len(evaluate_img) + 1e-19) * patch_count
                    # 列入 pix 评估 -> 用 lab_pixes+pre_pixes 赋权 （对 mf0.5 和 mf2 不公平，但实现起来简单）
                    count_pix = [
                        v['lab_pixes'][(metric.only, )].sum()+v['pre_pixes'][(metric.only, )].sum() for v in validates.values()
                    ]
                    evaluate_pix = [
                        v[metric.__name__] * c for c, v in zip(count_pix, validates.values())
                    ]
                    logs[f'{metric.__name__}_pix'] = sum(evaluate_pix) / (sum(count_pix) + 1e-19) * patch_count
                if self.verbose:
                    iterator.set_postfix_str(format_logs(logs, patch_count))
            # 解决最后一个老朋友
            stack_label = np.stack(labels, axis=0)
            stack_predict = np.stack(predicts, axis=0)
            predicts.clear()
            labels.clear()
            validates[name] = {
                'lab_pixes': stack_label.sum(axis=(0, 2, 3)),
                'pre_pixes': stack_predict.sum(axis=(0, 2, 3)),
            }
            for metric in self.metrics:
                validates[name][metric.__name__] = metric.calculate(stack_predict, stack_label).item()
            del stack_label
            del stack_predict
            # 最终统计
            for metric in self.metrics:
                # 只有 only 类型中囊括 lab 的， 才列入 img 评估
                evaluate_img = [
                    v[metric.__name__] for v in validates.values()
                    if v['lab_pixes'][(metric.only, )].sum() > 0
                ]
                logs[metric.__name__] = sum(evaluate_img) / (len(evaluate_img) + 1e-19) * patch_count
                # 列入 pix 评估 -> 用 lab_pixes+pre_pixes 赋权 （对 mf0.5 和 mf2 不公平，但实现起来简单）
                count_pix = [
                    v['lab_pixes'][(metric.only, )].sum()+v['pre_pixes'][(metric.only, )].sum() for v in validates.values()
                ]
                evaluate_pix = [
                    v[metric.__name__] * c for c, v in zip(count_pix, validates.values())
                ]
                logs[f'{metric.__name__}_pix'] = sum(evaluate_pix) / (sum(count_pix) + 1e-19) * patch_count
            if self.verbose:
                iterator.set_postfix_str(format_logs(logs, patch_count))

        # logs[self.loss.__name__] /= patch_count
        logs = {name: value / patch_count for name, value in logs.items()}
        self.on_epoch_end(logs)
        return logs


class ValidEpoch(MyEpoch):
    def __init__(self, model, loss, metrics: tuple = ()):
        super().__init__(model, loss, metrics)
        self.classes = hyp['dataset.class_num']

    def on_epoch_start(self, *args):
        self.model.eval()

    def batch_update(self, image, label):
        with torch.no_grad():
            prediction = self.model(image)
            loss = self.loss(prediction, label)
        return loss, prediction

    # @override
    def run(self, dataloader, epoch):
        self.on_epoch_start(epoch)
        logs = {self.loss.__name__: 0}
        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not self.verbose) as iterator:
            labels = []
            predicts = []
            validates = {}
            patch_count = 0
            for image, label, names in iterator:
                image, label = image.to(self.device), label.to(self.device),
                loss, predict = self.batch_update(image, label)
                # loss, predict = torch.tensor(1), label.clone()
                del image
                # 评估损失
                patch_num = len(names)
                patch_count += patch_num
                logs[self.loss.__name__] += loss.item() * patch_num
                # 按图块评估评价函数
                label = label.cpu().detach().numpy()
                predict = predict.cpu().detach().numpy()
                predict = one_hot_numpy(np.argmax(predict, axis=1), self.classes).transpose((0, 3, 1, 2))
                for i, name in enumerate(names):
                    # print(name)
                    if bool(labels):
                        # 进来新玩意了， 终结老朋友， 如果老朋友存在的话
                        stack_label = np.stack(labels, axis=0)
                        stack_predict = np.stack(predicts, axis=0)
                        predicts.clear()
                        labels.clear()
                        validates[name] = {
                            'lab_pixes': stack_label.sum(axis=(0, 2, 3)),
                            'pre_pixes': stack_predict.sum(axis=(0, 2, 3)),
                        }
                        for metric in self.metrics:
                            validates[name][metric.__name__] = metric.calculate(stack_predict, stack_label).item()
                        del stack_label
                        del stack_predict
                    # 进来老朋友了，先暂存起来
                    labels.append(label[i, :, :, :])
                    predicts.append(predict[i, :, :, :])
                    continue
                # 实时统计
                for metric in self.metrics:
                    # 只有 only 类型中囊括 lab 的， 才列入 img 评估
                    evaluate_img = [
                        v[metric.__name__] for v in validates.values()
                        if v['lab_pixes'][(metric.only, )].sum() > 0
                    ]
                    logs[metric.__name__] = sum(evaluate_img) / (len(evaluate_img) + 1e-19) * patch_count
                    # 列入 pix 评估 -> 用 lab_pixes+pre_pixes 赋权 （对 mf0.5 和 mf2 不公平，但实现起来简单）
                    count_pix = [
                        v['lab_pixes'][(metric.only, )].sum()+v['pre_pixes'][(metric.only, )].sum() for v in validates.values()
                    ]
                    evaluate_pix = [
                        v[metric.__name__] * c for c, v in zip(count_pix, validates.values())
                    ]
                    logs[f'{metric.__name__}_pix'] = sum(evaluate_pix) / (sum(count_pix) + 1e-19) * patch_count
                if self.verbose:
                    iterator.set_postfix_str(format_logs(logs, patch_count))
            # 解决最后一个老朋友
            stack_label = np.stack(labels, axis=0)
            stack_predict = np.stack(predicts, axis=0)
            predicts.clear()
            labels.clear()
            validates[name] = {
                'lab_pixes': stack_label.sum(axis=(0, 2, 3)),
                'pre_pixes': stack_predict.sum(axis=(0, 2, 3)),
            }
            for metric in self.metrics:
                validates[name][metric.__name__] = metric.calculate(stack_predict, stack_label).item()
            del stack_label
            del stack_predict
            # 最终统计
            for metric in self.metrics:
                # 只有 only 类型中囊括 lab 的， 才列入 img 评估
                evaluate_img = [
                    v[metric.__name__] for v in validates.values()
                    if v['lab_pixes'][(metric.only, )].sum() > 0
                ]
                logs[metric.__name__] = sum(evaluate_img) / (len(evaluate_img) + 1e-19) * patch_count
                # 列入 pix 评估 -> 用 lab_pixes+pre_pixes 赋权 （对 mf0.5 和 mf2 不公平，但实现起来简单）
                count_pix = [
                    v['lab_pixes'][(metric.only, )].sum()+v['pre_pixes'][(metric.only, )].sum() for v in validates.values()
                ]
                evaluate_pix = [
                    v[metric.__name__] * c for c, v in zip(count_pix, validates.values())
                ]
                logs[f'{metric.__name__}_pix'] = sum(evaluate_pix) / (sum(count_pix) + 1e-19) * patch_count
            if self.verbose:
                iterator.set_postfix_str(format_logs(logs, patch_count))

        # logs[self.loss.__name__] /= patch_count
        logs = {name: value / patch_count for name, value in logs.items()}
        self.on_epoch_end(logs)
        return logs