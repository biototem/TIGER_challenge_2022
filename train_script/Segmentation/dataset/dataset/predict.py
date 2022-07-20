from typing import List, Tuple, Dict, Any
import numpy as np
from torch.utils.data.dataloader import Dataset as StandardDataset

from .base import Dataset
from ..cropper import GroupCropper
from ..formatter import weights_formatter


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

    def pget(self, grid: dict) -> np.ndarray:
        image = self.cropper.get(**grid)
        if self.use_format:
            return weights_formatter.image(image)
        else:
            return image

    def get_image(self, name: str):
        return self.cropper.get_image(name)

    def get_label(self, name: str, use_box: bool = False):
        label = self.cropper.get_label(name)
        assert label is not None, 'Not ready for full-image-predict!'
        return label


class Subset(StandardDataset):
    def __init__(self, name: str, grids: dict, pset: Dataset):
        self.name = name
        # self.grids = list(map(lambda grid: dict(name=name, **grid), grids))
        self.grids = grids
        self.pset = pset

    def __len__(self) -> int:
        return len(self.grids)

    def __getitem__(self, index) -> Tuple[Dict[str, Any], np.ndarray]:
        grid = dict(name=self.name, **self.grids[index])
        return self.grids[index], self.pset.pget(grid)

    def get_HW(self) -> Tuple[int, int]:
        return self.pset.get_HW(self.name)

    def get_image(self) -> np.ndarray:
        return self.pset.get_image(self.name)

    def get_label(self) -> np.ndarray:
        return self.pset.get_label(self.name)
