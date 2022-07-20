import csv

import torch
from torch.utils.data import DataLoader

from basic import config, source, join
from utils import Assert, Timer
from model import ModelConfig
from dataset import Trainset
from .valid import valid


def do_valid(T: Timer = None):
    # 比赛评分规则：2、6 两种基质类型合并为一类评价
    # 1 和 基质 的dice值取平均
    # 其它类型忽略
    # 关于“逐图片还是逐像素”，赛方没有给出一个明确解释，但从例程中可以观察到
    # 代码是逐样本运行的，也就是说，在一次测试过程中，只有一张tif大图被传入程序之中
    # 因此我们只需要考虑“逐大图”即可 -> 但我们没有大图，所以对我们来说这可能是全部数据集
    model_names = ['loss', 'dice', 'dice_tumor', 'dice_stroma']
    # root = '~/output/latest'
    root = '~/output/competation-v2.0'
    models = [(name, f'{name}.pth') for name in model_names]
    names = source['group']['available'].keys()
    # tumor - stroma - mean - dice
    ts_scores = {}
    if T: T.track(' -> initing dataset')
    dataset_dict = {
        name: Trainset(
            names=[name],
            length=-1,
            cropper_type='simple',
            return_label=False,
            return_subset=False,
            return_pos=True,
            cropper_rotate='list[0]',
            cropper_scaling='list[1]',
        ) for name in names
    }
    with torch.no_grad():
        for name, model in models:
            if T: T.track(f' -> initing model -> {name}')
            m = torch.load(join(root, model))
            m = m.eval()
            if T: T.track(f' -> validating')
            scores = valid(m, dataset_dict, T.tab(), visual=join(root, name))
            ts_scores[name] = scores

    with open(join('~/output/evaluate.csv'), 'w') as f:
        writer = csv.writer(f, delimiter=';')
        keys = ['all'] + [k for k in scores.keys() if k != 'all']
        writer.writerow(('epoch', *keys))
        for name in model_names:
            values = name, *(ts_scores[name][k]['dice'] for k in keys)
            writer.writerow(values)


def do_valid_all(T: Timer = None):
    # 比赛评分规则：2、6 两种基质类型合并为一类评价
    # 1 和 基质 的dice值取平均
    # 其它类型忽略
    # 关于“逐图片还是逐像素”，赛方没有给出一个明确解释，但从例程中可以观察到
    # 代码是逐样本运行的，也就是说，在一次测试过程中，只有一张tif大图被传入程序之中
    # 因此我们只需要考虑“逐大图”即可 -> 但我们没有大图，所以对我们来说这可能是全部数据集
    EPOCH = 20
    root = '~/output/latest'
    models = [f'auto-save-epoch-{i}.pth' for i in range(1, EPOCH + 1)]
    names = source['group']['valid'].keys()
    # tumor - stroma - mean - dice
    ts_scores = []
    if T: T.track(' -> initing dataset')
    dataset_dict = {
        name: Trainset(
            names=[name],
            length=-1,
            cropper_type='simple',
            return_label=False,
            return_subset=False,
            return_pos=True,
            cropper_rotate='list[0]',
            cropper_scaling='list[1]',
        ) for name in names
    }
    with torch.no_grad():
        for i, model in enumerate(models):
            if T: T.track(f' -> initing model epoch-{i}')
            m = torch.load(join(root, model))
            m = m.eval()
            if T: T.track(f' -> validating')
            scores = valid(m, dataset_dict, T.tab())
            ts_scores.append(scores)

    with open(join('~/output/evaluate.csv'), 'w') as f:
        writer = csv.writer(f, delimiter=';')
        keys = ['all'] + [k for k in scores.keys() if k != 'all']
        writer.writerow(('epoch', *keys))
        for epoch in range(EPOCH):
            values = epoch + 1, *(ts_scores[epoch][k]['dice'] for k in keys)
            writer.writerow(values)
