from basic import config
from .cross_entropy import CrossEntropy
from .dice import Dice, GenDice, SoftDice
from .focal import Focal
from .iou import SoftIoU
from .special import Special
# from .lovasz import lovasz_softmax_onehot
from .lovasz_confidence import Lovasz
from .custom import Combine, LabelSmoothLimitedCrossEntropy, LCEDice


def loss():
    assert config['model.loss'] is not None, 'please offer loss in config.yaml'
    name = config['model.loss']
    if name == 'cross_entropy':
        return CrossEntropy()
    if name == 'special':
        return Special()
    if name == 'smooth_limited_ce':
        return LabelSmoothLimitedCrossEntropy()
    if name == 'dice':
        return Dice()
    if name == 'gen_dice':
        return GenDice()
    if name == 'iou':
        return SoftIoU(n_classes=config['dataset.class_num'])
    if name == 'lovasz':
        return Lovasz()
    if name.startswith('combine:'):
        a, b, c = (float(v) for v in name[8:].split(','))
        return Combine(a, b, c)
    if name.startswith('lce_dice:'):
        a = float(name[9:])
        return LCEDice(a)
    # 剩下这三个损失函数 暂时没有迁移到 confidence 上
    # if name == 'soft_dice':
    #     return SoftDice()
    # if name == 'focal':
    #     return Focal()
    # if name == 'lovasz':
    #     return lovasz_softmax_onehot()
    raise NotImplementedError(f'Loss [{name}] not implemented!')
