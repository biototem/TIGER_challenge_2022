import numpy as np

from basic import config
from utils import Drawer
from dataset import Trainset


def draw(argmaxed: np.ndarray, data_key: str, dataset: Trainset, root: str):
    # 由 Drawer 约定，预测图可视化需要以 one-hot 编码呈现，故此处需要转换
    classes_num = config['dataset.class_num']
    predict = np.eye(classes_num, dtype=bool)[argmaxed]
    # 以同样逻辑加载 label
    l, u, r, d = dataset.loaders[data_key].box
    label = dataset.loaders[data_key].label({
        'x': (l + r) // 2,
        'y': (u + d) // 2,
        'w': (r - l),
        'h': (d - u),
        'degree': 0,
        'scaling': 1,
    })
    label = np.eye(classes_num, dtype=bool)[label]
    # 加载 image
    image = dataset.loaders[data_key].image({
        'x': (l + r) // 2,
        'y': (u + d) // 2,
        'w': (r - l),
        'h': (d - u),
        'degree': 0,
        'scaling': 1,
    })
    # 可视化
    drawer = Drawer(root)
    # First for image/label/predict -> visual as origin
    drawer.name(data_key)
    drawer.image(image)
    drawer.label(label)
    drawer.predict(predict)
    # Second for label-predict at tumor -> visual specially
    drawer.name(f'{data_key}-tumor')
    drawer.inter(predict[:, :, (1, 3)], label[:, :, (1, 3)])
    # Third for label-predict at stroma -> visual specially
    drawer.name(f'{data_key}-stroma')
    drawer.inter(predict[:, :, (6, 2)], label[:, :, (6, 2)])




















def draw_old(argmaxed: np.ndarray, data_key: str, dataset: Trainset, root: str):
    # 由 Drawer 约定，预测图可视化需要以 one-hot 编码呈现，故此处需要转换
    classes_num = config['dataset.class_num']
    predict = np.eye(classes_num)[argmaxed]
    # 以同样逻辑加载 label
    l, u, r, d = dataset.loaders[data_key].box
    label = dataset.loaders[data_key].label({
        'x': (l + r) // 2,
        'y': (u + d) // 2,
        'w': (r - l),
        'h': (d - u),
        'degree': 0,
        'scaling': 1,
    })
    label = np.eye(classes_num)[label]
    # 加载 image
    image = dataset.loaders[data_key].image({
        'x': (l + r) // 2,
        'y': (u + d) // 2,
        'w': (r - l),
        'h': (d - u),
        'degree': 0,
        'scaling': 1,
    })
    # 可视化
    drawer = Drawer(root)
    drawer.name(data_key)
    drawer.predict(predict)
    drawer.label(label)
    drawer.predict_label(predict, label)
    drawer.image_predict_label(image, predict, label)
