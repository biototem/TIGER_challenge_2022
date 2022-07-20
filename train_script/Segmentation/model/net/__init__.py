import segmentation_models_pytorch as smp
from basic import config, join
from utils import Assert

from .net24 import Net as Net24
from .van import Van


def net():
    Assert.not_none(name=config['model.name'], classes=config['dataset.class_num'], activate=config['model.active'])
    name = config['model.name'].lower()

    our_params = {
        'in_dim': 3,
        'out_dim': config['dataset.class_num'],
        'activation': config['model.active'],
        'use_auxiliary': config['model.auxiliary'],
    }
    if name == 'net21':
        raise NotImplementedError('现在还不是时候')
    if name == 'net24':
        return Net24(**our_params)
    if name == 'van':
        return Van(**{**our_params, 'pretrained': join('~/env/van_base_828.pth.tar')})

    return smp.Unet(
        encoder_name=config['model.name'],
        encoder_weights=config['model.weights'],
        classes=config['dataset.class_num'],
        activation=config['model.active'],
        decoder_attention_type='scse',
    )
