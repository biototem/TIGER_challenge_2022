from .image import ImageLoader
from .numpy import NumpyLoader
from .slide import SlideLoader
from .shape import ShapeLoader

LOADERS = {
    'image': ImageLoader,
    'numpy': NumpyLoader,
    'slide': SlideLoader,
    'shape': ShapeLoader,
}


class Loader(object):
    def __init__(self, part_id: str, info: dict, in_memory: bool = False):
        # info = {
        #     'name': name,               # name 是数据源标识
        #     'image': path['image'],     # image 表示图片位置（jpg,png,tif,svs等）
        #     'image_loader': 'image',    # 图片识别类型：image/numpy/slide
        #     'label': path['label'],     # label 表示标签位置（png,npy,shape等）
        #     'label_loader': 'image',    # 标签识别类型：image/numpy/shape
        #     'sight': (0, 0, w, h),      # sight 表示可视区域在数据源中的坐标(l, u, w, h)
        #     'box': (0, 0, w, h),        # box 表示采集区域在数据源中的坐标(l, u, w, h)
        #     'size': (w, h),             # size 表示数据源的尺寸
        #     'zoom': 1,                  # zoom 表示数据源的标准缩放倍率（zoom==2表示此图分辨率更高、需要缩小）
        # }
        self.part_id = part_id
        self.name = info['name']
        # 数据源的信息到此为止，不再向下传递
        self.origin_sight = info['sight']
        self.origin_box = info['box']
        self.origin_size = info['size']
        self.origin_image_zoom = info['image_zoom']
        self.origin_label_zoom = info['label_zoom']

        # 数据加载器的核心方法就是 image 和 label，它们有着不同的复杂实现
        # 图片加载器
        image_loader = LOADERS[info['image_loader']]
        self.image = image_loader(info['image'], in_memory=in_memory)
        # 标签加载器
        label_loader = LOADERS[info['label_loader']]
        self.label = label_loader(info['label'], in_memory=in_memory)

    # 以下属性均为数据集的外显属性，已经滤除了数据源差异
    # sight 和 box 向 label 对齐 -> 因为它们表示标签区域的大小
    # size 向 image 对齐 -> 因为它表示数据源的大小
    @property
    def sight(self):
        return tuple(int(v / self.origin_label_zoom) for v in self.origin_sight)

    @property
    def box(self):
        return tuple(int(v / self.origin_label_zoom) for v in self.origin_box)

    @property
    def size(self):
        return tuple(int(v / self.origin_image_zoom) for v in self.origin_size)

