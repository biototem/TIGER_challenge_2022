import torch
import torch.optim as optim
import segmentation_models_pytorch as smp
from torch.nn.parallel.data_parallel import DataParallel

from basic import config
from ..net import Net24, Van


def optimizer(model: torch.nn.Module) -> optim.Optimizer:
    """
    这个方法很快就要修改掉了,但现在先用着吧
    TODO: 上面这句话应该是 2021.11 写的，但是到现在都还没改，我甚至不确定它还能不能修改……
    以下四个关键词：
    unet： 使用 segmentation_models_pytorch 原装网络模型，以及继承自 浩森 的编解码器独立学习率
    wt： 使用文泰模型，后来强行植入了编解码器独立学习率
    DataParallel: 这个是在 .188 服务器上训练时，受限于显存， batch_size 太小，担心影响训练效果，因此用 DataParallel 包装为多 GPU 训练
                  作为历史遗留代码，这个判定条件可以忽视（为什么不直接删掉它呢？）
    van: 使用 van 网络模型与参数
    """
    if model.__class__ == smp.Unet:
        return unet_optimizer(model)
    elif model.__class__ == Net24:
        return wt_optimizer(model)
    elif model.__class__ == Van:
        return van_optimizer(model)
    elif model.__class__ == DataParallel:
        return optimizer(next(model.children()))
    else:
        raise NotImplemented('Model not allowed!')


def unet_optimizer(model: torch.nn.Module) -> optim.Optimizer:
    # scheduler 用来调整learning_rate，一般在epoch中执行一次
    # 具体的调整方案取决于上文定义

    # use different learning-rate for encoder and decoder -> encoder is more stable than decoder

    params = [
        # 先试试直接锁掉 encoder
        {
            'params': model.encoder.parameters(),
            'lr': float(config['model.lr.encoder']),
            'weight_decay': 0,
        },
        {
            'params': model.decoder.parameters(),
            'lr': float(config['model.lr.decoder']),
            'weight_decay': 1e-1 / sum(map(lambda p: p.numel(), model.decoder.parameters())),
        },
        {
            'params': model.segmentation_head.parameters(),
            'lr': float(config['model.lr.head']),
            'weight_decay': 1 / sum(map(lambda p: p.numel(), model.segmentation_head.parameters())),
        },
    ]

    name = config['model.optimizer']
    # 'adam' -> normal optimizer <-> 'radam' -> stable but slow than 'adam' <-> 'sgd' -> the simplest optimizer
    if name == 'adam':
        return torch.optim.Adam(params=params, lr=1, eps=1e-17)
    elif name == 'radam':
        return torch.optim.RAdam(params=params, lr=1, eps=1e-17)
    elif name == 'sgd':
        return torch.optim.SGD(params=params, lr=1)
    else:
        raise NotImplementedError(f'{name} not implemented')


def wt_optimizer(model: torch.nn.Module) -> optim.Optimizer:
    params = [
        {
            'params': model.m1.parameters(),
            'lr': 0.2 * float(config['model.lr.wt_net']),
        },
        {
            'params': model.m2.parameters(),
            'lr': float(config['model.lr.wt_net']),
            'weight_decay': 0.1 * 1 / sum(map(lambda p: p.numel(), model.m2.parameters())),
        },
    ]
    name = config['model.optimizer']
    if name == 'adam':
        return torch.optim.Adam(params=params, eps=1e-8)
    elif name == 'radam':
        return torch.optim.RAdam(params=params, eps=1e-8)
    elif name == 'sgd':
        return torch.optim.SGD(params=params, lr=1)
    else:
        raise NotImplementedError(f'{name} not implemented')


def van_optimizer(model: torch.nn.Module) -> optim.Optimizer:
    params = model.parameters()
    name = config['model.optimizer']
    lr = float(config['model.lr.van'])
    if name == 'adam':
        return torch.optim.Adam(params=params, lr=lr, eps=1e-8)
    elif name == 'radam':
        return torch.optim.RAdam(params=params, lr=lr, eps=1e-8)
    elif name == 'sgd':
        return torch.optim.SGD(params=params, lr=lr)
    else:
        raise NotImplementedError(f'{name} not implemented')
