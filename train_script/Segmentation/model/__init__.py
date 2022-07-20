from .net import net
from .loss import loss
from .metric import metric
from .optimizer import optimizer
from .scheduler import scheduler

__all__ = ['ModelConfig']


class ModelConfig:
    net = net
    loss = loss
    metrics = metric
    optimizer = optimizer
    scheduler = scheduler
