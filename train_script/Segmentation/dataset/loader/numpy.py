from utils import Rotator


def gen(func: callable, arg: str) -> callable:
    def proxy():
        print('loading ... ')
        return func(arg)
    return proxy


class NumpyLoader(object):
    def __init__(self, path: str, zoom: float, **kwargs):
        self.path = path
        self.zoom = zoom
        register(path, gen(Rotator.NumpyCropper, path))

    def __call__(self, grid: dict):
        loader = query(self.path)
        # loader = Rotator.ImageCropper(self.path)
        return loader.get(
            site=(grid['x'], grid['y']),
            size=(grid['w'], grid['h']),
            degree=grid['degree'],
            scale=grid['scaling'] * self.zoom,
        )
