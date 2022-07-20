import abc
import math
import numpy as np
import PIL.Image as Image

"""
设计目的：
    本代码希望实现 通用的、便捷的、可扩展的 旋转切图采样工具
    它的功能是针对 '某一大图' 实现 '定点、定尺寸、定角度、定缩放' 切图
    因此，它总计需要 3-5 个固定参数，分别是： 
        大图（数据源）、定点（位置）、定尺寸（宽高）、定角度（角度制）、定缩放（缩放功能仅供参考）
    由于设计结构的问题，这些参数的提供方式不尽相同，详见下节

设计结构：
    库中包含一个抽象类 RotateCropper 和三个实现类 ImageRotateCropper、NumpyRotateCropper、SlideRotateCropper
    三个实现类均在创建时初始化相应的数据源，而它们的数据源各不相同：
        ImageRotateCropper 数据源是Image对象，参数接收 一个 filepath 或 一个 image 对象
        NumpyRotateCropper 数据源是Numpy对象，参数接收 一个 array 数组
        SlideRotateCropper 数据源是Slide对象，参数接收 一个 filepath
    它们在 slide 方法中统一将数据源转换为 numpy 数组返回，调用 slide 的抽象类中则包含具体的切图算法
    如果您需要扩展更多的数据源形式，只需继承 RotateCropper 并实现 slide 方法就可以了，也许您的需求不必定义 __init__
    
    抽象类 RotateCropper 中包含两个方法，分别是 __getitem__ 和 get，前者实现了一个语法糖：切片语法
    get 接收 2-4 个参数，分别用于表示（位置、宽高、角度、缩放），其中 角度=0 和 缩放=1 是可选参数
    __getitem__ 接收 2-4 个参数，分别用于表示（X起止坐标、Y起止坐标、角度、缩放），其中 角度=0 和 缩放=1 是可选参数
    您可以这样使用它们：
                get: -> cp.get((x0, y0), (w, h), degree, scale)
        __getitem__: -> cp[x1:x2, y1:y2, degree, scale]
    也许您会产生这样的疑惑：既然本代码返回的是一个倾斜矩形，那么 __getitem__ 所谓的起止坐标和原图是如何对应的？
    事实上，您可以这样理解：
        将 [x1:x2, y1:y2] 固定在原图上，假设给定参数 degree=15， scale=2
        让 [x1:x2, y1:y2] 逆时针旋转 15 度，然后放大 2 倍，所得的矩形就是目标采样区域
        将这个采样区域摆正，就得到了我们需要的、倾斜的采样结果
        因此看起来，图像像是被 顺时针旋转了15度，并缩小至原来的一半

最后，让我们来试试下面的例程
"""


# 例程
def test():
    import openslide
    import matplotlib.pyplot as plt
    # 首先定义我们的数据源，两个路径均指向 211 服务器上的图片
    img_path = '/cache/visual/H1804782 1 HE_image.jpg'
    image = Image.open(img_path)
    array = np.asarray(image)
    tif_path = '/media/totem_disk/totem/guozunhu/Project/kidney_biopsy/tiff_data/tif/H1804782 1 HE/H1804782 1 HE_Wholeslide_默认_Extended.tif'
    # 以不同的格式传给数据源
    cp1 = ImageRotateCropper(filepath=img_path)
    cp2 = ImageRotateCropper(image=image)
    cp3 = NumpyRotateCropper(array=array)
    # 特别要注意的是，本例程中只有 slide 指向了原图 tif，因此它的截图永远完整
    cp4 = SlideRotateCropper(filepath=tif_path)

    # 例如：希望以 (x=1000, y=500) 为中心截定边长为 (w=100, h=200) 的矩形
    # slide： 原图
    patch0 = cp1.slide(950, 400, 100, 200)
    # __getitem__： 顺时针旋转 15度
    patch1 = cp1[950:1050, 400:600, 15]
    # get: 逆时针旋转 15度， 缩放 2倍
    patch2 = cp1.get(site=(1000, 500), size=(100, 200), degree=-15, scale=2)

    # 我们将截图起点的坐标做成参数
    w, h = 100, 150
    for cp, (px, py) in zip([cp1, cp2, cp3, cp4], [(0, 0), (0, 0), (0, 0), (3949 * 2, 3605 * 2)]):
        # slide 方法是用来获取数据的，我们也可以用它直接获得未旋转的原图
        patch0 = cp.slide(px, py, w, h)
        # 请注意，我们截图的位置选在了边缘，这意味着旋转过后，有一部分数据将会是黑的
        patch1 = cp[px:px+w, py:py+h, 15]
        patch2 = cp.get(site=(px+w//2, py+h//2), size=(w, h), degree=15)
        # 在这张截图中，我们选择了更大的 scale，因此它的边缘将更加不完整
        patch3 = cp.get(site=(px+w//2, py+h//2), size=(w, h), degree=15, scale=2)
        # 展示
        plt.subplot(1, 4, 1)
        plt.imshow(patch0)
        plt.subplot(1, 4, 2)
        plt.imshow(patch1)
        plt.subplot(1, 4, 3)
        plt.imshow(patch2)
        plt.subplot(1, 4, 4)
        plt.imshow(patch3)
        plt.show()
        # 再将这些图片旋转回去
        patch_1 = rotate(patch1, -15)
        patch_2 = rotate(patch2, -15, padding=np.array([0, 0, 255]))
        patch_3 = rotate(patch3, -15, padding=np.array([0, 0, 255]))
        patch_4 = rotate(patch3, -15, padding=np.array([0, 0, 255]), scale=0.5)
        # 展示
        plt.subplot(1, 4, 1)
        plt.imshow(patch_1)
        plt.subplot(1, 4, 2)
        plt.imshow(patch_2)
        plt.subplot(1, 4, 3)
        plt.imshow(patch_3)
        plt.subplot(1, 4, 4)
        plt.imshow(patch_4)
        plt.show()


# ----------------------------------库------------------------------------


__all__ = ['SlideRotateCropper', 'ImageRotateCropper', 'NumpyRotateCropper', 'rotate', 'get_size_rate']


class RotateCropper(object):
    """
    abstract class for RotateCroppers defining the interface
    the implements should realize the image-source with numpy
    """

    def __getitem__(self, items):
        """
        :param items: slice(x), slice(y), degree, scale
        :return: a numpy-array
        """
        assert isinstance(items, tuple) and len(items) in (2, 3, 4), \
            'the slice should be [slice(x), slice(y), degree, scale]! ' \
            'please check ' + ', '.join(map(str, items))
        x_slice, y_slice = items[:2]
        assert isinstance(x_slice, slice) and isinstance(y_slice, slice) \
               and x_slice.step is None and y_slice.step is None \
               and x_slice.start is not None and y_slice.start is not None \
               and x_slice.stop is not None and y_slice.stop is not None, \
               'slice(x) and slice(y) must be [a:b]!'
        degree = 0 if len(items) < 3 else items[2]
        scale = 1 if len(items) < 4 else items[3]
        x_center = (x_slice.start + x_slice.stop) / 2
        y_center = (y_slice.start + y_slice.stop) / 2
        return self.get(
            site=(x_center, y_center),
            size=(x_slice.stop - x_slice.start, y_slice.stop - y_slice.start),
            degree=degree,
            scale=scale,
        )

    def get(self, site, size, degree: float = 0, scale: float = 1):
        """
        :param site: position(x, y) for the center of target area
        :param size: int or (int, int) for the width and height of target area
        :param degree: a float[-180:+180] for the image rotated-degree in anti-clock
        :param scale: a float > 0 seems like sample-level
        :return: a numpy-array of rotation-result
        """
        # count the window
        x_center, y_center = site
        w, h = (size, size) if isinstance(size, (int, float)) else size
        r = degree * np.pi / 180
        sina, cosa = np.sin(r), np.cos(r)
        w1 = math.ceil((w * abs(cosa) + h * abs(sina)) * scale)
        h1 = math.ceil((h * abs(cosa) + w * abs(sina)) * scale)
        x1, y1 = round(x_center - w1 / 2), round(y_center - h1 / 2)
        # get the img-array -> need to be implement
        img = self.slide(x1, y1, w1, h1)
        # warp_affine -> https://blog.csdn.net/qq_40939814/article/details/117966835
        # build map-matrix
        x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
        x_grid = x_grid - w / 2
        y_grid = y_grid - h / 2
        x_index = ((cosa * x_grid + sina * y_grid) * scale + w1 / 2).round().astype(np.int32)
        y_index = ((cosa * y_grid - sina * x_grid) * scale + h1 / 2).round().astype(np.int32)
        # use the numpy-broadcast
        return img[
            np.clip(y_index, 0, h1-1),
            np.clip(x_index, 0, w1-1),
        ]

    @abc.abstractmethod
    def slide(self, x, y, width, height):
        raise NotImplemented('please implement it in sub-class of RotateCropper')


class SlideRotateCropper(RotateCropper):
    def __init__(self, filepath: str, level: int = 0):
        import openslide
        self._slide = openslide.OpenSlide(filename=filepath)
        self.level = level

    def slide(self, x, y, width, height):
        image = self._slide.read_region(
            location=(x, y),
            level=self.level,
            size=(width, height)
        )
        return np.asarray(image)


class ImageRotateCropper(RotateCropper):
    def __init__(self, filepath: str = None, image: Image.Image = None):
        assert not (filepath is None and image is None), 'please provide the image-source!'
        assert not (filepath is not None and image is not None), 'please provide only one image-source!'
        if filepath is not None:
            self._slide = Image.open(fp=filepath)
        else:
            self._slide = image

    def slide(self, x, y, width, height):
        image = self._slide.crop((x, y, x + width, y + height))
        return np.asarray(image)


class NumpyRotateCropper(RotateCropper):
    def __init__(self, array: np.ndarray):
        assert len(array.shape) >= 2, 'your array must contain at least 2 dims'
        self._slide = array

    def slide(self, x, y, width, height):
        _x = 0 if x < 0 else x
        _y = 0 if y < 0 else y
        array = self._slide[_y:y + height, _x:x + width]
        # 当取值点距边缘太近时，补0填充
        if array.shape[:2] != (height, width):
            # 左上边界填充值
            ux = _x - x
            uy = _y - y
            # 右下边界填充值
            dx = width - array.shape[1] - ux
            dy = height - array.shape[0] - uy
            following = [(0, 0)] * len(array.shape[2:])
            array = np.pad(array, ((uy, dy), (ux, dx), *following), 'constant')
        return array


def rotate(array: np.ndarray, degree: float, scale: float = 1, padding: np.ndarray = np.array(0)):
    """
    :param array: Numpy(height, width, ...) to be rotated
    :param degree: anti-clock -> float[-180, +180]
    :param scale: float > 0
    :param padding: a default vector to init result
    :return: Numpy(HEIGHT, WIDTH, ...) -> a bigger numpy
    """
    assert array is not None and len(array.shape) >= 2, 'array should be an image-matrix, got {}'.format(array.shape if array else None)
    # init a temp
    r = degree * np.pi / 180
    sina, cosa = np.sin(r), np.cos(r)
    h, w = array.shape[:2]
    w1 = math.ceil((w * abs(cosa) + h * abs(sina)) / scale)
    h1 = math.ceil((h * abs(cosa) + w * abs(sina)) / scale)
    # temp = np.zeros(shape=(h1, w1) + array.shape, dtype=array.dtype)
    # temp[:, :] = padding

    # warp_affine -> https://blog.csdn.net/qq_40939814/article/details/117966835
    # build map-matrix
    x_grid, y_grid = np.meshgrid(np.arange(w1), np.arange(h1))
    x_grid = x_grid - w1 / 2
    y_grid = y_grid - h1 / 2
    x_index = ((cosa * x_grid + sina * y_grid)*scale + w / 2).round().astype(np.int32)
    y_index = ((cosa * y_grid - sina * x_grid)*scale + h / 2).round().astype(np.int32)
    # mapping
    condition = (x_index >= 0) & (x_index < w) & (y_index >= 0) & (y_index < h)
    # x_index[np.logical_not(condition)] = 0
    # y_index[np.logical_not(condition)] = 0
    # print(w, h, w1, h1, array.shape, condition.shape, x_index[condition].max(), y_index[condition].max())
    # print(array.shape, condition.shape, padding.shape)
    # expand dims and use the numpy-broadcast
    condition = np.expand_dims(condition, tuple(range(2, len(array.shape))))
    padding = np.expand_dims(padding, tuple(range(len(array.shape) - 2)))
    return np.where(
        condition,
        array[
            np.clip(y_index, 0, h-1),
            np.clip(x_index, 0, w-1),
        ],
        padding
    )


def get_size_rate(degree: float):
    radius = degree * np.pi / 180
    sina = np.sin(radius)
    cosa = np.cos(radius)
    return abs(sina) + abs(cosa)


if __name__ == '__main__':
    test()
