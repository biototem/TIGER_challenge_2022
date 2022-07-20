from utils import Rotator


class ImageLoader(object):
    def __init__(self, path: str, in_memory: bool = False):
        self.path = path
        self.in_memory = in_memory
        self.cache = None

    def load(self) -> Rotator.ImageCropper:
        if self.in_memory:
            if self.cache is None:
                self.cache = Rotator.ImageCropper(self.path)
            return self.cache
        else:
            return Rotator.ImageCropper(self.path)

    def __call__(self, grid: dict):
        loader = self.load()
        return loader.get(
            site=(grid['x'], grid['y']),
            size=(grid['w'], grid['h']),
            degree=grid['degree'],
            scale=grid['scaling'],
        )
