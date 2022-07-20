import sys

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from basic import config
from utils import Timer, TorchMerger, Drawer
from model import ModelConfig


def valid(model, dataset_dict, T: Timer, visual: str = False):
    # loss_func = ModelConfig.loss()
    # loss_func.to(0)
    # metric_func = ModelConfig.metrics()
    # metric_func.to(0)
    patch_info_set = {}

    # 使用 loader
    for name, dataset in dataset_dict.items():
        loader = DataLoader(
            dataset=dataset,
            batch_size=config['train.batch_size'],
            num_workers=config["train.num_workers"]
        )
        l, u, r, d = dataset.loaders[name].box
        with TorchMerger(class_num=8, kernel_size=512, kernel_steep=6).with_shape(W=r-l, H=d-u) as merger:
            with tqdm(loader, desc='valid', file=sys.stdout, disable=True) as iterator:
                for images, grids in iterator:
                    images = images.to(0)
                    predicts = model(images)
                    merger.set(predicts, grids)
                predict = merger.tail()

        label = dataset.loaders[name].label({
            'x': (l + r) // 2,
            'y': (u + d) // 2,
            'w': (r - l),
            'h': (d - u),
            'degree': 0,
            'scaling': 1,
        })
        if visual:
            drawer = Drawer(visual)
            drawer.name(name)
            drawer.label(predict)
            visual_label = np.eye(8)[label]
            drawer.predict_label(predict, visual_label)
            image = dataset.loaders[name].image({
                'x': (l + r) // 2,
                'y': (u + d) // 2,
                'w': (r - l),
                'h': (d - u),
                'degree': 0,
                'scaling': 1,
            })
            drawer.image_predict_label(image, predict, visual_label)

        predict = np.argmax(predict, axis=2)

        tumor_pr = (predict == 1).sum()
        tumor_gt = (label == 1).sum()
        tumor_inter = ((predict == 1) & (label == 1)).sum()
        tumor_dice = 2 * tumor_inter / (tumor_pr + tumor_gt + 1e-17)
        stroma_pr = ((predict == 2) | (predict == 6)).sum()
        stroma_gt = ((label == 2) | (label == 6)).sum()
        stroma_inter = (
                ((predict == 2) | (predict == 6)) &
                ((label == 2) | (label == 6))
        ).sum()
        stroma_dice = 2 * stroma_inter / (stroma_pr + stroma_gt + 1e-17)
        patch_info_set[name] = {
            'pixes': (r - l) * (d - u),
            'tumor_pr': tumor_pr,
            'tumor_gt': tumor_gt,
            'tumor_inter': tumor_inter,
            'tumor_dice': tumor_dice,
            'stroma_pr': stroma_pr,
            'stroma_gt': stroma_gt,
            'stroma_inter': stroma_inter,
            'stroma_dice': 2 * stroma_inter / (stroma_pr + stroma_gt + 1e-17),
            'dice': (tumor_dice + stroma_dice) / 2,
        }
    sums = {
        'pixes': 0,
        'tumor_pr': 0,
        'tumor_gt': 0,
        'tumor_inter': 0,
        'tumor_dice': 0,
        'stroma_pr': 0,
        'stroma_gt': 0,
        'stroma_inter': 0,
        'stroma_dice': 0,
        'dice': 0,
    }
    for score_dict in patch_info_set.values():
        for k, v in score_dict.items():
            sums[k] += v
    sums['tumor_dice'] = 2 * sums['tumor_inter'] / (sums['tumor_pr'] + sums['tumor_gt'] + 1e-17)
    sums['stroma_dice'] = 2 * sums['stroma_inter'] / (sums['stroma_pr'] + sums['stroma_gt'] + 1e-17)
    sums['dice'] = (sums['tumor_dice'] + sums['stroma_dice']) / 2
    patch_info_set['all'] = sums
    return patch_info_set
