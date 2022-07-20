import abc

import numpy as np
from segmentation_models_pytorch.encoders import get_preprocessing_fn as fn
import albumentations as A

__all__ = ['trans_formatter']


def trans_formatter(cfg: dict):
    # No formatter 并不是真的啥也不做，而是只做维度转换，不做数值处理
    if cfg['type'] == 'no':
        return NoFormatter()
    elif cfg['type'] == 'simple':
        return SimpleFormatter()
    elif cfg['type'] == 'norm':
        return NormFormatter()
    elif cfg['type'] == 'smp':
        return PretrainedFormatter(cfg)


class Formatter(abc.ABC):
    def image(self, image: np.ndarray) -> np.ndarray:
        raise NotImplemented

    def label(self, image: np.ndarray) -> np.ndarray:
        raise NotImplemented


class NoFormatter(object):
    # 由于数据采集的过程中少不了负数索引的运算，而torch不支持numpy的负索引
    # 所以这里必须显式的调用 copy 函数，令 numpy 从懒汉式转入饿汉式，从而消除负索引
    # return image, label
    @staticmethod
    def image(image: np.ndarray) -> np.ndarray:
        return image.transpose((2, 0, 1)).astype(np.float32).copy()

    @staticmethod
    def label(label: np.ndarray) -> np.ndarray:
        return label.transpose((2, 0, 1)).astype(np.uint8).copy()


class SimpleFormatter(object):
    # 由于数据采集的过程中少不了负数索引的运算，而torch不支持numpy的负索引
    # 所以这里必须显式的调用 copy 函数，令 numpy 从懒汉式转入饿汉式，从而消除负索引
    # return image, label
    @staticmethod
    def image(image: np.ndarray) -> np.ndarray:
        return (image.transpose((2, 0, 1)) / 255).astype(np.float32).copy()

    @staticmethod
    def label(label: np.ndarray) -> np.ndarray:
        return label.transpose((2, 0, 1)).astype(np.uint8).copy()


class NormFormatter(object):
    std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
    mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))

    # imagenet 的 norm 规范化
    @staticmethod
    def image(image: np.ndarray) -> np.ndarray:
        return ((image / 255 - NormFormatter.mean) / NormFormatter.std).transpose((2, 0, 1)).astype(np.float32).copy()

    @staticmethod
    def label(label: np.ndarray) -> np.ndarray:
        return label.transpose((2, 0, 1)).astype(np.uint8).copy()


class PretrainedFormatter(object):
    def __init__(self, cfg: dict):
        self.fmt = A.Lambda(image=fn(encoder_name=cfg['name'], pretrained=cfg['weights']))

    def image(self, image: np.ndarray) -> np.ndarray:
        r = self.fmt(image=image)
        return r['image'].transpose((2, 0, 1)).astype(np.float32)

    def label(self, label: np.ndarray) -> np.ndarray:
        r = self.fmt(mask=label)
        return r['mask'].transpose((2, 0, 1)).astype(np.uint8)


# if not hyp['model.name'].lower().startswith('net') and hyp['model.weights']:
#     weights_formatter = pretrained_formatter()
# else:
#     weights_formatter = simple_formatter
