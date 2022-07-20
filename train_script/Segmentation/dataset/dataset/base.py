from typing import List, Tuple
import numpy as np
from torch.utils.data.dataloader import Dataset as StandardDataset

from utils import Timer
from ..cropper import Cropper
from ..formatter import weights_formatter


class Dataset(StandardDataset):
    def __init__(self, cropper: Cropper, names_list: List[str], use_format=True, T: Timer = None):
        if names_list:
            cropper.filter(names_list)
        self.cropper = cropper.begin(T=T)
        self.samples = self.sample()
        self.use_format = use_format

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, item: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.get(self.samples[item])

    def sample(self) -> List[dict]:
        return self.cropper.sample()

    def get(self, grid: dict) -> Tuple[np.ndarray, np.ndarray]:
        return self.format(*self.cropper.get(**grid))

    def format(self, image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.use_format:
            return image, label
        image = weights_formatter.image(image)
        label = weights_formatter.label(label)
        return image, label

    # def get_sight(self, name: str, use_format: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    #     image = self.cropper.get_image(name)
    #     label = self.cropper.get_label(name)
    #     if use_format:
    #         return self.format(image, label)
    #     else:
    #         return image, label

    def get_HW(self, name: str):
        info = self.cropper.get_info(name)
        _, _, w, h = info['sight']
        return h, w

    def get_box(self, name: str):
        return self.cropper.get_info(name)['box']
