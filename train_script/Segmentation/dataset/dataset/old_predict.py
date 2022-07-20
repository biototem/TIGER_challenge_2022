import abc
from typing import List, Tuple, Dict, Any
import numpy as np
from torch.utils.data.dataloader import Dataset as StandardDataset

from .base import Dataset
from ..cropper import GroupCropper


class PredictSet(Dataset):
    def __init__(self,
                 name_list: List[str] = None,
                 use_migrate: bool = False,
                 use_confidence: bool = False,
                 use_format: bool = True,
                 use_box: bool = True,
                 ):
        super().__init__(GroupCropper(use_migrate, use_confidence, use_box), name_list, use_format=use_format)

    def __getitem__(self, item: int) -> Exception:
        raise NotImplemented('Never predict in datasource-loader')

    def __iter__(self) -> iter:
        for group in self.samples:
            yield Subset(group['name'], group['grids'], self)


class ImageSet(StandardDataset, abc.ABC):
    @abc.abstractmethod
    def get_HW(self) -> Tuple[int, int]:
        raise NotImplementedError('Need to be inherit')

    # @abc.abstractmethod
    # def __iter__(self):
    #     raise NotImplementedError('Need to be inherit')


class Subset(ImageSet):
    def __init__(self, name: str, grids: dict, dataset: Dataset):
        self.name = name
        # self.grids = list(map(lambda grid: dict(name=name, **grid), grids))
        self.grids = grids
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.grids)

    def __getitem__(self, index) -> Tuple[Dict[str, Any], np.ndarray]:
        grid = dict(name=self.name, **self.grids[index])
        return self.grids[index], self.dataset.get(grid)[0]

    def get_HW(self) -> Tuple[int, int]:
        return self.dataset.get_HW(self.name)

    def get_sight(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.dataset.get_sight(self.name)
