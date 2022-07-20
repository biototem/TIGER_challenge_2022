from typing import List
import numpy as np

from basic import config, source
from utils import Assert
from .adaptor import Adaptor
from .interface import Dataset
from .cropper import NoCropper, SimpleCropper, DenseCropper, ActiveCropper, RandomCropper
from .loader import Loader


class Trainset(Dataset):
    def __init__(self, **kwargs):
        # 解析参数
        p = {
            # 参数有三种约定：
            # 1. 应当由参数传递
            # 2. 应当默认
            # 3. 应当由配置文件给定
            'source': source,
            'names': source['group']['train'],
            # 决定类型的总数
            'class_num': config['dataset.class_num'] or 2,
            # 决定训练时 image、numpy 的存储位置；但 slide 始终在硬盘中，shape始终在内存中
            'in_memory': config['dataset.in_memory'] or False,
            # 决定每 epoch 的长度
            'length': config['dataset.length'] or 100,
            # 决定是否随机打乱
            'shuffle': True,
            # 数据增强方法
            'transfer': config['dataset.transfer'],
            # 数据标准化方法
            'formatter': config['dataset.formatter'],
            # 决定切图类型：不切图、均匀切图、密铺切图、随机切图、采样均衡切图
            'cropper_type': config['cropper.type'],
            'cropper_rotate': config['cropper.rotate'] or 'list[0]',
            'cropper_scaling': config['cropper.scaling'] or 'list[1]',
            # 决定切图区域的限变裕量（为正数：可在region外多取；为负数：可在region内少取）
            'cropper_margin': config['cropper.margin'] or 0,
            # 决定返回类型（是否携带坐标信息、标签等）
            'return_subset': config['dataset.return.subset'] or False,
            'return_label': config['dataset.return.label'] or False,
            'return_pos': config['dataset.return.pos'] or False,
        }
        p.update(**kwargs)
        Assert.not_none(**p)

        # 加载器用于加载数据，并统一接口格式
        self.loaders = {
            part_id: Loader(part_id, info, in_memory=p['in_memory']) for
            part_id, info in source['data'].items()
            if info['name'] in p['names']
        }
        # 适配器用于解析参数状态，提供琐碎功能支持
        self.adaptor = Adaptor(**p)
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
        self.shuffle = p['shuffle']
        self.length = p['length']

    def __len__(self) -> int:
        # 训练数据集在每次调用 len 时进行采样
        self.samples = self.sample()
        return len(self.samples)

    def sample(self) -> List[dict]:
        result = []
        if self.length == -1:
            for name, loader in self.loaders.items():
                # 切图器根据每张图的尺寸特征生成采样锚点
                # 当 num=-1 时，cropper 将根据图片本身的大小计算采样数，而不限制每张图的采样数
                result += self.cropper.sample(loader, num=-1)
            if self.shuffle:
                np.random.shuffle(result)
        else:
            while len(result) < self.length:
                for name, loader in self.loaders.items():
                    # 切图器根据每张图的尺寸特征生成采样锚点
                    # 当 num=-1 时，cropper 将根据图片本身的大小计算采样数，而不限制每张图的采样数
                    result += self.cropper.sample(loader, num=-1)
            if self.shuffle:
                np.random.shuffle(result)
            result = result[:self.length]
        return result

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
        # 最后返回结果
        result = (image, )
        if self.adaptor['return_label']:
            result += (label, )
        if self.adaptor['return_pos']:
            result += (grid, )
        return result if len(result) > 1 else image
