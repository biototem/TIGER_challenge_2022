import abc
from typing import List

from ..loader import Loader


class Cropper(abc.ABC):
    def sample(self, loader: Loader, num: int) -> List[dict]:
        raise NotImplemented
