import numpy as np
import json
import cv2
import imagesize

from basic import join, config, os
from utils import Assert, Timer, Drawer


def build_data(visual: bool = False, T: Timer = None):

    with open(join('~/resource/manual/image_uses.json')) as f:
        group = json.load(f)

    if T: T.track(f' -> using group {group}')
    # cache_root 主要用来存放label.shape，此处不需要，另，本项目使用png数据源，因此不需要mpp
    # Assert.not_none(
    #     cache_root=config['build.cache_root'],
    #     target_mpp=config['cropper.target_mpp']
    # )
    drawer = Drawer(root=config['build.visual_root'])
    if visual:
        Assert.not_none(
            visual_root=config['build.visual_root'],
            fill=config['visual.color.fill'],
            outline=config['visual.color.outline'],
        )
        if T: T.track(f' -> visual at {config["build.visual_root"]}')

    lib = {}
    for name, info in group.items():
        if T: T.track(f' -> building {name}')
        if info['tag'] == 'BROKEN':
            if T: T.track(f' -> broken image jumped')
            continue
        # color changed to new path
        info['image'] = join('~/resource/data/color_normalized', os.path.basename(info['image']))
        # 一张图片可能会分成数个子块，并封装入不同的数据集，这些信息将保存在data_lib中
        # 但此比赛项目不含此需求
        part_id = name
        # 获取图片宽高信息
        w, h = imagesize.get(info['image'])
        lib[part_id] = {
            'name': name,               # name 是数据源标识
            'image': info['image'],     # image 表示图片位置（jpg,png,tif,svs等）
            'image_loader': 'image',    # 图片识别类型：image/numpy/slide
            'image_zoom': 1,            # zoom 表示数据源的标准缩放倍率（zoom==2表示此图分辨率更高、需要缩小）
            'label': info['label'],     # label 表示标签位置（png,npy,shape等）
            'label_loader': 'image',    # 标签识别类型：image/numpy/shape
            'label_zoom': 1,            # zoom 表示数据源的标准缩放倍率（zoom==2表示此图分辨率更高、需要缩小）
            'sight': [0, 0, w, h],      # sight 表示可视区域在数据源中的坐标(l, u, w, h)
            'box': [0, 0, w, h],        # box 表示采集区域在数据源中的坐标(l, u, w, h)
            'size': [w, h],             # size 表示数据源的尺寸
            'tag': info['tag'],
        }
        # 数据可视化 - 按下采样倍率执行
        if visual:
            if T: T.track(f' -> visual image {name}')
            image = cv2.imread(info['image'])
            label = cv2.imread(info['label'])[:, :, 0]
            label = np.eye(8)[label]
            if info['tag'] == 'BROKEN': continue
            drawer.name(name)
            drawer.image(image=image, box=None)
            drawer.label(label=label, box=None)
            drawer.image_label(image=image, label=label, box=None)
    return lib
