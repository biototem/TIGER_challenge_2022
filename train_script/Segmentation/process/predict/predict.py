import sys
import numpy as np
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from basic import config
from utils import TorchMerger
from dataset import Trainset as Dataset


def predict(model: Module, data_key: str, dataset: Dataset):
    dataset.sample()
    loader = DataLoader(
        dataset=dataset,
        batch_size=config['train.batch_size'],
        num_workers=config["train.num_workers"]
    )
    l, u, r, d = dataset.loaders[data_key].box
    with TorchMerger(class_num=8, kernel_size=512, kernel_steep=6).with_shape(W=r - l, H=d - u) as merger:
        with tqdm(loader, desc='predict', file=sys.stdout, disable=True) as iterator:
            for images, grids in iterator:
                grids['x'] -= l
                grids['y'] -= u
                images = images.to('cuda:0')
                predicts = model(images)
                merger.set(predicts, grids)
            merged = merger.tail()
    return np.argmax(merged, axis=2)
