import torch

from basic import config
from segmentation_models_pytorch.losses import DiceLoss


class StandardDice(DiceLoss):
    def __init__(self, ignore: int = None):
        super().__init__(
            mode='multiclass',
            classes=config['dataset.class_num'],
            log_loss=False,
            from_logits=True,
            smooth=0.0,
            ignore_index=ignore,
            eps=1e-17,
        )
        self.__name__ = 'standard-dice-loss'

    def forward(self, pr, gt, *args):
        gt = torch.argmax(gt, dim=1)
        return super().forward(pr, gt)
