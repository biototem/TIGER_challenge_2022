import sys
from typing import Tuple

import torch
import os
import yaml
import csv
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from basic import join, source, config
from utils import Timer, TorchMerger
from dataset import Trainset
from utils.table.my_table import Table
from .calculate import do_calculate
from .evaluate import do_evaluate


class Helper(object):
    """
    用于训练过程中的日志记录
    """
    def __init__(self, model: torch.nn.Module, T: Timer = None):
        # 指定输出目录 output/target
        self.root = join(config["output.root"], config['output.target'])
        self.root_cal = join(self.root, 'calculate')
        self.root_evl = join(self.root, 'evaluate')
        self.root_vis = join(self.root, 'visual')
        # 创建相关目录
        os.makedirs(self.root, exist_ok=True)
        os.makedirs(self.root_cal, exist_ok=True)
        os.makedirs(self.root_evl, exist_ok=True)
        os.makedirs(self.root_vis, exist_ok=True)
        # 保存参数文件到本地 - 备份 yml 至 output 下，以便我们知道训练结果是用哪套参数跑出来的 -> 但我们还是得手写一遍
        with open(join(self.root, "CONFIG.yaml"), mode="w", encoding="utf-8") as fp:
            yaml.dump(dict(config), fp)

        self.model = model
        self.log_headers = []
        self.T = T
        self.log_state = False

        if T: T.track(' -> initializing validate dataset')
        self.data_dict = get_data_dict()

    def process(self, epoch: int, visual: bool = False):
        """
        训练辅助流程：
        1. 保存权重
        2. 权重预测
        3. 预测结果评估
        4. 预测结果可视化
        5. 日志记录（tensorboard日志、评估记录、excel日志、最佳模型日志）
        """
        if self.T: self.T.track(f' -> epoch {epoch + 1} training done!')
        if self.T: self.T.track(f' -> saving model weights')
        torch.save(self.model, join(self.root, f'auto-save-epoch-{epoch + 1}.pth'))

        # TODO: 我怀疑后面的代码有 bug，可能导致了显存泄漏
        return

        if self.T: self.T.track(f' -> predicting')
        # 预测、评估 和 日志记录
        table = self.calculate(epoch, visual=visual)

        score, wdice, dice26, dice134 = self.evaluate(epoch, table)

        self.log(epoch, {
            'score': score,
            'wdice': wdice,
            'dice26': dice26,
            'dice134': dice134,
        })

    def calculate(self, epoch: int, visual: bool = False):
        # 获得预测矩阵
        table = do_calculate(
            model=self.model,
            data_dict=self.data_dict,
            T=self.T and self.T.tab(),
            visual=visual,
            visual_root=join(self.root_vis, f'{epoch + 1}')
        )
        # 保存原始预测信息
        table.save(path=join(self.root_cal, f'{epoch + 1}.txt'))
        # 返回预测表
        return table

    def evaluate(self, epoch: int, data_table: Table):
        # 加载统计信息
        # data_table = Table.load(path=join(self.root_cal, f'{epoch + 1}.txt'), dtype=list)
        # 获得评估表
        score_table = do_evaluate(data_table)

        # 保存评估信息
        score_table.save(path=join(self.root_evl, f'{epoch + 1}.txt'))
        # 返回主要评估指标
        return score_table['all', ('score', 'wdice', 'dice26', 'dice134')].data.tolist()

    def log(self, epoch: int, logs: dict):
        # 必须在 with 语法内才执行日志功能
        if not self.log_state:
            return

        for name, value in logs.items():
            self.logger.add_scalar(name, value, epoch + 1)
            if name not in self.log_headers:
                self.log_headers.append(name)
        if epoch == 0:
            line = 'epoch', *self.log_headers
            self.writer.writerow(line)
        line = epoch + 1, *('%.2f' % logs[name] for name in self.log_headers)
        self.writer.writerow(line)
        self.log_file.flush()

    def __enter__(self):
        self.log_state = True
        self.logger = SummaryWriter(
            log_dir=join(f'~/cache/tensorboard/{config["output.target"]}'),
            comment=config['output.target']
        )
        self.log_file = open(join(self.root, config['output.log']), 'w')
        self.writer = csv.writer(self.log_file, delimiter=';')

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.log_state = False
        if self.T: self.T.track(f' -> ending training')
        self.logger.close()
        self.log_file.close()
        return False


def get_data_dict():
    return {
        name: Trainset(
            names=[name],
            length=-1,
            cropper_type='simple',
            shuffle=False,
            transfer={'level': 'off'},
            return_label=False,
            return_subset=False,
            return_pos=True,
            cropper_rotate='list[0]',
            cropper_scaling='list[1]',
        # ) for name in source['group']['dev_test']
        ) for name in source['group']['valid']
    }
