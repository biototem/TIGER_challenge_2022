from typing import List, Tuple
import numpy as np

from basic import InterfaceProxy, config, source
from utils import Assert
from .adaptor import Adaptor
from .adaptor.transfer import transfer
from .interface import Dataset
from .cropper import ScalingSpinCropper, SpinCropper, SimpleCropper, ScalingSimpleCropper, ActiveCropper
from .loader import Loader


class Trainset(Dataset):
    def __init__(self, **kwargs):
        # 解析参数
        p = {
            # 应当由参数传递
            # 应当默认
            'source': source,
            'names': source['group']['train'].keys(),
            # 应当由配置文件给定
            # 决定类型的总数
            'class_num': config['dataset.class_num'] or 2,
            # 决定训练时 image、numpy 的存储位置；但 slide 始终在硬盘中，shape始终在内存中
            'in_memory': config['dataset.in_memory'] or False,
            # 决定每 epoch 的长度
            'length': config['dataset.length'] or 100,
            # 数据增强方法
            'transfer': config['dataset.transfer'],
            # 数据标准化方法
            'formatter': config['dataset.formatter'],
            # 决定切图类型：不切图、均匀切图、密铺切图、随机切图、采样均衡切图
            'cropper_type': config['cropper.type'],
            'cropper_rotate': config['cropper.rotate'] or 0,
            'cropper_scaling': config['cropper.scaling'] or 1,
            # 决定切图区域（相对原始数据的坐标）
            'cropper_region': config['cropper.region'] or (),
            # 决定返回类型（是否携带坐标信息、标签等）
            'return_subset': config['dataset.return.subset'] or False,
            'return_label': config['dataset.return.label'] or False,
            'return_pos': config['dataset.return.pos'] or False,
        }
        p.update(**kwargs)
        Assert.not_none(**p)

        # 加载器用于加载数据，并统一接口格式
        self.loaders = {name: Loader(source['data'][name]) for name in p['names']}
        # 适配器用于解析参数状态，提供琐碎功能支持
        self.adaptor = Adaptor(p)
        # 切图器用于生成锚点列，这也是采样算法的核心
        if p['cropper_type'] == 'no':
            self.cropper = NoCropper(self.adaptor)
        elif p['cropper_type'] == 'simple':
            self.cropper = SimpleCropper(self.adaptor)
        elif p['cropper_type'] == 'dense':
            self.cropper = DenseCropper(self.adaptor)
        elif p['cropper_type'] == 'random':
            self.cropper = RandomCropper(self.adaptor)
        elif p['cropper_type'] == 'active':
            self.cropper = ActiveCropper(self.adaptor)
        else:
            raise AttributeError(f'Illegal param -> cropper_type:{p["cropper_type"]}')
        self.samples = None
        self.length = p['length']

    def __len__(self) -> int:
        # 训练数据集在每次调用 len 时进行采样
        self.samples = self.sample()
        return len(self.samples)

    def sample(self) -> List[dict]:
        result = []
        while len(result) < self.length:
            for name, loader in self.loaders:
                # 切图器根据每张图的尺寸特征生成采样锚点
                # 当 num=-1 时，cropper 将根据图片本身的大小计算采样数，而不限制每张图的采样数
                result += self.cropper.sample(name, loader, num=-1)
        np.random.shuffle(result)
        return result[:self.length]

    def __getitem__(self, item):
        # 确定采样信息
        return self.get(self.samples[item])

    def get(self, grid):
        # 取得数据源
        loader = self.loaders[grid['name']]
        # 取得数据
        image = loader.image(grid)
        label = loader.label(grid) if self.adaptor['return_label'] else None
        # 训练时需要数据增强
        image, label = self.adaptor.transfer(image, label)
        # 然后进行数据标准化
        image = self.adaptor.format_image(image)
        label = self.adaptor.format_label(label)
        # 最后返回结果 -> 总是按顺序返回这三样，但后两者可能用不到，那就置None
        return image, label, grid


'''
        self.transfer = p['transfer']
        self.length = p['length']
        if p['use_active']:
            cropper = ActiveCropper(
                weights=config['dataset.active_weight'],
                scaling=config['dataset.train.scaling'],
                use_migrate=p['use_migrate'],
                use_confidence=p['use_confidence'],
                use_box=p['use_box'],
            )
        elif p['use_spin'] and p['use_scale']:
            cropper = ScalingSpinCropper(
                scaling=config['dataset.train.scaling'],
                use_migrate=p['use_migrate'],
                use_confidence=p['use_confidence'],
                use_box=p['use_box'],
            )
        elif p['use_spin'] and not p['use_scale']:
            # SpinCropper 尚未实现 use_box=False
            cropper = SpinCropper(p['use_migrate'], p['use_confidence'])
        elif not p['use_spin'] and not p['use_scale']:
            # SimpleCropper 尚未实现 use_box=False
            cropper = SimpleCropper(p['use_migrate'], p['use_confidence'])
        else:
            # SimpleCropper 尚未实现 use_box=False
            cropper = ScalingSimpleCropper(config['dataset.train.scaling'], p['use_migrate'], p['use_confidence'])
        super().__init__(cropper, p['name_list'], use_format=p['use_format'])
        self.label_rates = dict((name, crop_info['label_rate']) for name, crop_info in cropper.source.items())
        self.strengthen_rate = config['dataset.train.strengthen_rate']

'''

