import time
from typing import Tuple
import traceback
import numpy as np
import cv2
import threading

from basic import hyp, InterfaceProxy
from utils import Canvas


class Painter(threading.Thread):
    def __init__(self, save: callable):
        super().__init__()
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 1000, 1000)
        cv2.moveWindow('image', -1, -1)
        cv2.setMouseCallback('image', self)
        # 管理全部图片
        self.source = []
        self.processing = False
        self.index = 0
        # 管理图片状态
        self.box = None
        self.image = None
        self.history = None
        # 管理绘画状态
        self.label = None
        self.temp = None
        self.draw_time = False
        self.lines = []
        # 保存回调
        self.save = save

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.run()
        if exc_type is not None or exc_val is not None or exc_tb is not None:
            traceback.print_exc()
        return True

    def add(self, name, image, label, box: Tuple[int, int, int, int]):
        l, u, r, d = box
        l = max(l, 0)
        u = max(u, 0)
        r = min(r, image.shape[1])
        d = min(d, image.shape[0])

        label = label.copy()
        label[:, :l, 0] = 0
        label[:u, :, 0] = 0
        label[:, r:, 0] = 0
        label[d:, :, 0] = 0
        self.source.append({
            'name': name,
            'box': (l, u, r, d),
            'history': [label.copy()],
            # 'image': Canvas().with_image(image).image[:, :, (2, 1, 0)].copy()
            'image': Canvas().with_image(image)
                             .draw_color(label[:, :, 1], hyp['visual.color.outline'][1])
                             .draw_border(label[:, :, 2], hyp['visual.color.outline'][2])
                             .draw_border(label[:, :, 3], hyp['visual.color.outline'][3])
                             .image[:, :, (2, 1, 0)].copy()
        })
        return self

    def run(self):
        while len(self.source) < 1:
            time.sleep(0.25)
        self.switch('current')
        while True:
            self.processing = False
            # 每次绘完图都需要重新展示e
            cv2.imshow('image', self.draw())
            key = cv2.waitKey(50) & 0xFF
            # s 表示保存
            if key == 115:
                print('saved')
                self.processing = True
                data = self.source.pop(self.index)
                self.save(data['name'], data['history'][-1])
                del data
                # 所有图片都保存了就退出
                if len(self.source) == 0:
                    break
                self.switch('current')
            # q 表示上一张
            if key == 113:
                print('switch to last')
                self.processing = True
                self.switch('last')
            # e 表示下一张
            if key == 101:
                print('switch to next')
                self.processing = True
                self.switch('next')
            # 27 是 ESC 的退出码，表示主动退出
            if key == 27:
                print('exit')
                break
        cv2.destroyWindow('image')
        # return [business['history'][-1] for business in self.source]

    def switch(self, flag: str):
        if flag == 'next':
            p = 1
        elif flag == 'last':
            p = -1
        elif flag == 'current':
            p = 1
            self.index -= 1
        else:
            raise RuntimeError('flag not exists')
        i = self.index + p
        while i < 0 or i >= len(self.source):
            i += p
            if i >= len(self.source):
                i = 0
            elif i < 0:
                i = len(self.source) - 1
            elif i == self.index:
                raise RuntimeError('no more images')
        self.index = i
        # print(f'painting image {self.source[i]["name"]}')
        cv2.setWindowTitle('image', self.source[i]["name"])
        self.box = self.source[i]['box']
        self.image = self.source[i]['image']
        self.label = self.source[i]['history'][-1].copy()
        self.history = self.source[i]['history']
        self.temp = np.zeros_like(self.image)
        self.draw_time = False
        self.lines.clear()

    def __call__(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            if not self.draw_time: return
            # print('line', x, y)
            x0, y0 = self.lines[-1]
            if (x - x0) ** 2 + (y - y0) ** 2 < 25: return
            self.lines.append((x, y))
            self.trace()
        elif event == cv2.EVENT_LBUTTONDOWN:
            # print('draw')
            self.lines.append((x, y))
            self.draw_time = time.time()
        elif event == cv2.EVENT_LBUTTONUP:
            # print('drawn')
            self.lines.append((x, y))
            self.paint()
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            print('clear')
            if len(self.history) > 1:
                self.history.pop()
            self.label = self.history[-1].copy()
            self.paint()

    def draw(self):
        painted = self.image.copy()
        mask = self.label.sum(axis=2).astype(bool)
        painted[~mask, :] //= 2
        mask = self.temp.astype(bool)
        painted[mask] = self.temp[mask]
        return painted

    def trace(self):
        x1, y1 = self.lines[-2]
        x2, y2 = self.lines[-1]
        cv2.line(self.temp, (x1, y1), (x2, y2), (0, 0, 255), 11)

    def paint(self):
        # 时间太短、长度太短的直接摒弃
        if not self.draw_time or time.time() - self.draw_time < 0.7 or len(self.lines) <= 4:
            self.draw_time = False
            self.lines.clear()
            self.temp[:, :, :] = 0
            return
        self.draw_time = False
        d1, x1, y1 = self.moveto(self.lines[0])
        d2, x2, y2 = self.moveto(self.lines[-1])
        d = {**d1, **d2}
        # 有划线的，总共分定四种情况
        # 自围
        if (x1 - x2) ** 2 + (y1 - y2) ** 2 < 400:
            pass
        # 围角
        elif 'x' in d and 'y' in d:
            self.lines.append((x2, y2))
            self.lines.append((d['x'], d['y']))
            self.lines.append((x1, y1))
        # 围边
        elif ('x' in d1 and 'x' in d2 and d1['x'] == d2['x']) or \
                ('y' in d1 and 'y' in d2 and d1['y'] == d2['y']):
            self.lines.append((x2, y2))
            self.lines.append((x1, y1))
        # 无效
        else:
            self.lines.clear()
            self.temp[:, :, :] = 0
            return
        # cv2.polylines(self.temp, [np.array(lines)], True, [0, 255, 0], 11)
        cv2.fillPoly(self.temp, [np.array(self.lines)], [255, 0, 0])
        self.lines.clear()
        mask = self.temp[:, :, 0].astype(bool)
        self.temp[:, :, :] = 0
        self.label[mask, 0] = 0
        self.history.append(self.label.copy())

    def moveto(self, p):
        b = self.box
        l = abs(p[0] - b[0])
        u = abs(p[1] - b[1])
        r = abs(p[0] - b[2])
        d = abs(p[1] - b[3])
        if min(l, u, r, d) > 100:
            return {}, p[0], p[1]
        if min(l, u, r, d) == l:
            return {'x': b[0]}, b[0], p[1]
        if min(l, u, r, d) == u:
            return {'y': b[1]}, p[0], b[1]
        if min(l, u, r, d) == r:
            return {'x': b[2]}, b[2], p[1]
        if min(l, u, r, d) == d:
            return {'y': b[3]}, p[0], b[3]


if __name__ == '__main__':
    # b = (100, 100, 500, 500)
    # img = np.load('/media/totem_disk/totem/jizheng/renal_tubule/datasource/lib_image/H1804782 1 HE.npy')
    # lbl = np.load('/media/totem_disk/totem/jizheng/renal_tubule/datasource/lib_label/H1804782 1 HE.npy')
    # print(img.shape, img[:, :, (2, 1, 0)].shape)
    # img = draw_box(img, box, 11, (0, 0, 255))
    Painter(save=InterfaceProxy.EMPTY).add(
        'H1804782 1 HE',
        np.load('/media/totem_disk/totem/jizheng/renal_tubule/cache/lib_image/H1804782 1 HE.npy'),
        np.load('/media/totem_disk/totem/jizheng/renal_tubule/cache/lib_label/H1804782 1 HE.npy'),
        (100, 100, 500, 700)
    ).add(
        'H1803134 1 HE',
        np.load('/media/totem_disk/totem/jizheng/renal_tubule/cache/lib_image/H1803134 1 HE.npy'),
        np.load('/media/totem_disk/totem/jizheng/renal_tubule/cache/lib_label/H1803134 1 HE.npy'),
        (100, 100, 500, 700)
    ).add(
        'H1519947 1 HE',
        np.load('/media/totem_disk/totem/jizheng/renal_tubule/cache/lib_image/H1519947 1 HE.npy'),
        np.load('/media/totem_disk/totem/jizheng/renal_tubule/cache/lib_label/H1519947 1 HE.npy'),
        (100, 100, 500, 700)
    ).start()
