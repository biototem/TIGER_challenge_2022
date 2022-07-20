import os

import numpy as np
import cv2
from typing import List, Tuple, Union

from basic import config, join
from .shape import SingleShape
from .model import one_hot_numpy


def classify(predict: np.ndarray):
    full_mask = predict.sum(axis=2) > 0.98
    if full_mask.sum() < 4:
        l, r, u, d = 0, 0, 0, 0
    else:
        ids = np.nonzero(full_mask)
        l, r, u, d = ids[1].min(), ids[1].max(), ids[0].min(), ids[0].max()
    classify_predict = one_hot_numpy(np.argmax(predict, axis=2), num=config['dataset.class_num'])
    classify_predict[~full_mask, :] = 0
    return (l, u, r, d), classify_predict


class Canvas(object):
    def __init__(self, line_width: int = 0):
        self.image = None
        self.line_width = line_width or config['visual.bord_width']
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(self.line_width, self.line_width))

    def with_shape(self, shape):
        self.image = np.zeros(shape=(*shape, 3), dtype=np.uint8)
        return self

    def with_image(self, image: np.ndarray):
        self.image = image.astype(np.uint8)
        return self

    def with_label(self, label: np.ndarray, colors: List):
        self.image = np.dot(label, np.array(colors)).astype(np.uint8)
        return self

    def _with_instance_colorful(self, instance: np.ndarray):
        """
        instance[h, w] -> int32 为实例图
        instance == -1 -> 边界
        instance == 0 -> 背景
        instance >= 1 -> 实例
        """
        pass

    def with_instance_gray(self, full: np.ndarray):
        self.image = np.zeros(shape=(*full.shape, 3), dtype=np.uint8)
        self.draw_color(full > 0, color=(100, 100, 100))
        self.draw_color(full == -1, color=(0, 255, 255))
        return self

    def with_instance(self, instance: np.ndarray):
        """
        instance[h, w] -> int32 为实例图
        instance == -1 -> 边界
        instance == 0 -> 背景
        instance >= 1 -> 实例
        """
        color_num = 0
        temp = np.zeros(shape=(*instance.shape, 3), dtype=np.uint8)
        # 边界涂 红
        temp[instance == -1, :] = [0, 255, 255]
        # 背景涂 灰
        temp[instance == 0, :] = [90, 25, 100]
        # 实例图 随机色
        n = instance.max()
        color_list = [i * 29 for i in range(n)]
        # color_list = [color_list[i * 8 % n + i * 8 // n] for i in range(n)]
        hue_list = [c % 120 for c in color_list]
        # saturation_list = [255 - c // 120 % 4 * 30 for c in color_list]
        saturation_list = [c // 120 for c in color_list]
        saturation_list = [{0: 255, 1: 160, 2: 200, 3: 120}[s % 4] for s in saturation_list]
        value_list = [255 for c in color_list]
        for i in range(n):
            # color_num = (314159269 * color_num + 453806245) % (1 << 31) % 480
            temp[instance == (i+1), :] = [hue_list[i], saturation_list[i], value_list[i]]
        self.image = cv2.cvtColor(temp, cv2.COLOR_HSV2RGB)
        return self

    def draw_border(self, label: np.ndarray, color: Tuple[int, int, int]):
        assert self.image.shape[:2] == label.shape, \
            'image shape {}, label shape {}'.format(self.image.shape, label.shape)
        mask = label.astype(bool).astype(np.uint8)
        mask -= cv2.erode(mask, self.kernel)
        self.image[mask.astype(bool), :] = np.asarray(color)
        return self

    def draw_color(self, mask: np.ndarray, color: Tuple[int, int, int]):
        self.image[mask.astype(bool), :] = np.asarray(color)
        return self

    def draw_mix(self, mask: np.ndarray, ratio: float, color: Tuple[int, int, int]):
        assert 0 < ratio < 1, f'ratio must be a float in (0, 1), got {ratio}'
        temp = self.image * (1 - ratio)
        temp += np.asarray(color) * ratio
        mask = mask.astype(bool)
        self.image[mask, :] = temp[mask, :]
        # 下面这行代码慢的离谱，肯定不能用
        # self.image[mask, :] = self.image[mask, :] * (1-ratio) + np.asarray(color) * ratio
        return self

    def draw_box(self, box: Tuple[int, int, int, int], color: Tuple[int, int, int]):
        color = np.asarray(color)
        l, u, r, d = box
        l = max(l, 0)
        u = max(u, 0)
        r = min(r, self.image.shape[1])
        d = min(d, self.image.shape[0])
        p = self.line_width
        self.image[u: d, l: l + p, :] = color
        self.image[u: d, r - p: r, :] = color
        self.image[u: u + p, l: r, :] = color
        self.image[d - p: d, l: r, :] = color
        return self

    def draw_contour(self, contour: Union[List[Tuple[int, int]], SingleShape], color: Tuple[int, int, int]):
        if isinstance(contour, SingleShape):
            contour = np.asarray(contour.sep_p(), dtype=np.int32)
        contour = np.asarray(contour, dtype=np.int32)
        cv2.drawContours(
            image=self.image,
            contours=[contour],
            contourIdx=-1,
            color=color,
            thickness=self.line_width,
        )
        return self

    def draw_txt(self, txt: str, pos: Tuple[int, int],  color: Tuple[int, int, int]):
        cv2.putText(
            img=self.image,
            text=txt,
            org=pos,
            # fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontFace=cv2.FONT_ITALIC,
            fontScale=0.5,
            color=color,
            thickness=1,
        )
        return self

    def draw_info(self, info, print_txt=False):
        for contour in info['contours']:
            # 绘制轮廓
            color = config['visual.color.fill'][2] if contour['base']['tp'] == 'normal' else config['visual.color.fill'][3]
            self.draw_contour(contour['contour'], color=color)
            if print_txt:
                # 文字喷涂
                txt = '%d' % int(contour['base']['pixes'])
                x, y = map(int, contour['center'])
                p = x - 15, y
                color = [0, 255, 255]
                self.draw_txt(txt, p, color=color)
                if contour['base']['rn_score'] < 0.5:
                    txt = 'warn-fp'
                    p = x - 30, y - 15
                    self.draw_txt(txt, p, color=color)
                if contour['base']['tp_score'] < 0.5:
                    txt = 'warn-fc'
                    p = x - 30, y + 15
                    self.draw_txt(txt, p, color=color)
        return self


class Drawer(object):
    def __init__(self, root: str):
        os.makedirs(name=root, exist_ok=True)
        self.root = root
        self.line_width = config['visual.bord_width']
        self.__name__ = 'drawer'

    def name(self, name: str):
        self.__name__ = name
        return self

    def image(self, image: np.ndarray, box: Tuple[int, int, int, int] = None):
        if box is None:
            h, w, _ = image.shape
            box = (0, 0, w, h)
        cv2.imwrite(
            join(self.root, f'{self.__name__}-image.jpg'),
            Canvas(self.line_width)
            .with_image(image)
            .draw_box(box, config['visual.color.box'])
            .image[:, :, (2, 1, 0)]
        )
        return self

    def label(self, label: np.ndarray, box: Tuple[int, int, int, int] = None):
        if box is None:
            h, w, _ = label.shape
            box = (0, 0, w, h)
        cv2.imwrite(
            join(self.root, f'{self.__name__}-label.jpg'),
            Canvas(self.line_width)
            .with_label(label, config['visual.color.fill'])
            .draw_box(box, config['visual.color.box'])
            .image[:, :, (2, 1, 0)]
        )
        return self

    def predict(self, predict: np.ndarray, box: Tuple[int, int, int, int] = None):
        if box is None:
            h, w, _ = predict.shape
            box = (0, 0, w, h)
        cv2.imwrite(
            join(self.root, f'{self.__name__}-predict.jpg'),
            Canvas(self.line_width)
            .with_label(predict, config['visual.color.fill'])
            .draw_box(box, config['visual.color.box'])
            .image[:, :, (2, 1, 0)]
        )
        return self

    def image_label(self, image: np.ndarray, label: np.ndarray, box: Tuple[int, int, int, int] = None):
        if box is None:
            h, w, _ = label.shape
            box = (0, 0, w, h)
        canvas = Canvas(self.line_width).with_image(image)
        for i in range(config['dataset.class_num']):
            if i == 0: continue
            canvas.draw_border(label[:, :, i], config['visual.color.outline'][i])
        cv2.imwrite(
            join(self.root, f'{self.__name__}-image-label.jpg'),
            canvas.draw_box(box, config['visual.color.box']).image[:, :, (2, 1, 0)]
        )
        return self

    def image_predict(self, image: np.ndarray, predict: np.ndarray, box: Tuple[int, int, int, int] = None):
        if box is None:
            h, w, _ = image.shape
            box = (0, 0, w, h)
        _, classify_predict = classify(predict)
        canvas = Canvas(self.line_width).with_image(image)
        for i in range(config['dataset.class_num']):
            if i == 0: continue
            canvas.draw_mix(classify_predict[:, :, i], 0.3, config['visual.color.fill'][i])
        cv2.imwrite(
            join(self.root, f'{self.__name__}-image-predict.jpg'),
            canvas.draw_box(box, config['visual.color.box']).image[:, :, (2, 1, 0)]
        )
        return self

    def predict_label(self, predict: np.ndarray, label: np.ndarray, box: Tuple[int, int, int, int] = None):
        if box is None:
            h, w, _ = label.shape
            box = (0, 0, w, h)
        canvas = Canvas(self.line_width).with_label(predict, config['visual.color.fill'])
        for i in range(config['dataset.class_num']):
            if i == 0: continue
            canvas.draw_border(label[:, :, i], config['visual.color.outline'][i])
        cv2.imwrite(
            join(self.root, f'{self.__name__}-predict-label.jpg'),
            canvas.draw_box(box, config['visual.color.box']).image[:, :, (2, 1, 0)]
        )
        return self

    def inter(self, predict: np.ndarray, label: np.ndarray, box: Tuple[int, int, int, int] = None):
        if box is None:
            h, w, _ = label.shape
            box = (0, 0, w, h)
        canvas = Canvas(self.line_width).with_shape(predict.shape[:2])
        # this method is visualized for no more than 3 types' inter
        assert len(predict.shape) == 3 and predict.dtype in [bool, int, np.uint8], 'method needs int & one-hot img'
        n = predict.shape[2]
        assert n==1 or n==2, f'Shape not match! Please Check input:{n}'
        if n == 1: grays = [255]
        else: grays = [255, 120]

        for i, gray in enumerate(grays):
            pr = predict[:, :, i]
            gt = label[:, :, i]
            tp = pr & gt
            fp = pr & ~gt
            fn = ~pr & gt
            # tp -> green
            canvas.draw_color(tp, (0, gray, 0))
            # fn -> blue
            canvas.draw_color(fn, (0, 0, gray))
            # fp -> yellow
            canvas.draw_color(fp, (gray, gray, 0))
        if n == 2:
            # wr -> red
            wr12 = predict[:, :, 1] & label[:, :, 0]
            wr21 = predict[:, :, 0] & label[:, :, 1]
            canvas.draw_color(wr12, (180, 0, 0))
            canvas.draw_color(wr21, (255, 0, 0))

        cv2.imwrite(
            join(self.root, f'{self.__name__}-inter.jpg'),
            canvas.draw_box(box, config['visual.color.box']).image[:, :, (2, 1, 0)]
        )
        return self

    def predict_label_iou(self, predict: np.ndarray, label: np.ndarray, box: Tuple[int, int, int, int] = None):
        if box is None:
            h, w, _ = label.shape
            box = (0, 0, w, h)
        # 直接叠起来,然后利用矩阵乘法,控制色值保证补超过255即可
        iou = np.concatenate([predict.astype(float), label.astype(float)], axis=2)
        cv2.imwrite(
            join(self.root, f'{self.__name__}-predict-label-iou.jpg'),
            Canvas(self.line_width)
            .with_label(iou, [
                # 在 iou 语境下, 背景和边界不予染色
                # 我们期待黄色表示 normal-TP, 紫色表示 atrophic-TP
                # 而 label, predict 则采用不同颜色加以区分
                # [0, 0, 0], [0, 0, 0], [155, 100, 0], [155, 0, 100],
                # [0, 0, 0], [0, 0, 0], [100, 155, 0], [100, 0, 155],
                # [0, 0, 0], [0, 0, 0], [55, 100, 0], [100, 0, 55],
                # [0, 0, 0], [0, 0, 0], [200, 155, 0], [155, 0, 200],
                [0, 0, 0], [0, 0, 0], [55, 100, 0], [55, 0, 200],
                [0, 0, 0], [0, 0, 0], [200, 155, 0], [200, 0, 55],
            ])
            .draw_box(box, config['visual.color.box'])
            .image[:, :, (2, 1, 0)]
        )

    def image_predict_label(self, image: np.ndarray, predict: np.ndarray, label: np.ndarray, box: Tuple[int, int, int, int] = None):
        if box is None:
            h, w, _ = label.shape
            box = (0, 0, w, h)
        _, classify_predict = classify(predict)
        canvas = Canvas(self.line_width).with_image(image)
        for i in range(config['dataset.class_num']):
            if i == 0: continue
            canvas.draw_mix(classify_predict[:, :, i], 0.3, config['visual.color.fill'][i])
        for i in range(config['dataset.class_num']):
            if i == 0: continue
            canvas.draw_border(label[:, :, i], config['visual.color.outline'][i])
        cv2.imwrite(
            join(self.root, f'{self.__name__}-image-predict-label.jpg'),
            canvas.draw_box(box, config['visual.color.box']).image[:, :, (2, 1, 0)]
        )
        return self

    def instance(self, instance: np.ndarray):
        cv2.imwrite(
            join(self.root, f'{self.__name__}-instance.jpg'),
            Canvas(self.line_width)
            .with_instance(instance)
            .image[:, :, (2, 1, 0)]
        )
        return self

    def info(self, info: dict):
        cv2.imwrite(
            join(self.root, f'{self.__name__}-info.jpg'),
            Canvas(line_width=3)
            .with_shape(shape=info['size'])
            .draw_info(info, print_txt=True)
            .image[:, :, (2, 1, 0)]
        )
        return self

    def image_info(self, image: np.ndarray, info: dict):
        cv2.imwrite(
            join(self.root, f'{self.__name__}-image-info.jpg'),
            Canvas(line_width=3)
            .with_image(image)
            .draw_info(info, print_txt=False)
            .image[:, :, (2, 1, 0)]
        )
        return self
