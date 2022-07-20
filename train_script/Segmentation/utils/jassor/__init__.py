from .magic_iter import magic_iter
from .rotate import rotate, get_size_rate, ImageRotateCropper, NumpyRotateCropper, SlideRotateCropper
from .merge import Merger

__all__ = [
    'magic_iter',
    'Rotator',
    # 'rotate',
    # 'ImageRotateCropper',
    # 'NumpyRotateCropper',
    # 'SlideRotateCropper',
    'Merger',
]


class Rotator(object):
    rotate = rotate
    size_rate = get_size_rate
    ImageCropper = ImageRotateCropper
    NumpyCropper = NumpyRotateCropper
    SlideCropper = SlideRotateCropper
    # TODO: 这个要加上
    ShapeCropper = None
