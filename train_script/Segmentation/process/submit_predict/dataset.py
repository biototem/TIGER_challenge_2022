from typing import List, Tuple
import numpy as np

from dataset.loader import ImageLoader


class Dataset(object):
    """
    注意，全图预测流程中的数据集仅仅是个封装傀儡，包括采样方法在内的一切参数都由外部给定
    """
    def __init__(
            self,
            image_loader: ImageLoader,
            grids: List[dict],
            formatter: callable,
    ):
        self.image_loader = image_loader
        self.grids = grids
        self.formatter = formatter

    def __len__(self) -> int:
        return len(self.grids)

    def __getitem__(self, item: int) -> Tuple[np.ndarray, dict]:
        grid = self.grids[item]
        image = self.image_loader(grid)
        image = self.formatter(image=image)
        return image, grid

    def image(self, grid: dict) -> np.ndarray:
        return self.image_loader(grid)
