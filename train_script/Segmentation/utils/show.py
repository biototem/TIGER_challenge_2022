from typing import Iterable
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from descartes import PolygonPatch
from shapely.geometry.base import BaseGeometry

from .shape import Shape


class PPlot(object):
    def __init__(self, name: str = 'pplt'):
        self.name = name
        self.imgs = []
        self.ttls = []
        plt.figure(name)

    def title(self, *titles):
        self.ttls.extend(titles)
        return self

    def add(self, *img):
        self.imgs.extend(img)
        return self

    def save(self, fname: str, dpi: int = 1000):
        self.__plot__()
        plt.savefig(fname=fname, dpi=dpi)
        plt.close(self.name)

    def show(self):
        self.__plot__()
        plt.show()

    def __plot__(self):
        n = len(self.imgs)
        i = int(n ** 0.5)
        j = (n + i - 1) // i
        for p in range(i):
            for q in range(j):
                k = p * j + q
                if k >= n:
                    break
                ax = plt.subplot(i, j, k + 1)
                if k < len(self.ttls) and self.ttls:
                    plt.title(self.ttls[k])
                # ax.axis('equal')
                # plt.imshow(self.imgs[k])
                self.__draw__(ax, self.imgs[k])

                plt.axis('equal')

    def __draw__(self, ax, img):
        if isinstance(img, np.ndarray):
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.imshow(img)
        elif isinstance(img, Shape):
            l, u, r, d = list(map(int, img.bounds))
            # print(l, u, r, d)
            # plt.xticks([l, r])
            # plt.yticks([u, d])
            ax.set_xticks([l, r])
            ax.set_yticks([u, d])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            # 变换矩阵: matrix = [xAx, xAy, yAx, yAy, xb, yb]
            # img = affine_transform(img, [1, 0, 0, -1, 0, d])
            if img:
                ax.add_patch(
                    PolygonPatch(img.geo, fc=[0, 0, 1, 0.5], ec=[0, 0, 1, 0.5], alpha=0.5, zorder=2)
                )
            ax.plot()
        elif isinstance(img, BaseGeometry):
            l, u, r, d = list(map(int, img.bounds))
            # print(l, u, r, d)
            # plt.xticks([l, r])
            # plt.yticks([u, d])
            ax.set_xticks([l, r])
            ax.set_yticks([u, d])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            # 变换矩阵: matrix = [xAx, xAy, yAx, yAy, xb, yb]
            # img = affine_transform(img, [1, 0, 0, -1, 0, d])
            ax.add_patch(
                PolygonPatch(img, fc=[0, 0, 1, 0.5], ec=[0, 0, 1, 0.5], alpha=0.5, zorder=2)
            )
            ax.plot()
        elif isinstance(img, Iterable):
            self.__draw__(ax, Polygon(img))
        elif isinstance(img, Image.Image):
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.imshow(img)
        else:
            raise NotImplementedError(f'{type(img)} is not supported!')


def draw_contours_tuple(*contours):
    """
    cv2.findContours 的 contours 格式: tuple[array] -> array(points, 1, (y, x))
    cv2.fillPoly 的 contours 格式: (batch, points, (y, x))
    """
    pplt = PPlot()
    for i, contour in enumerate(contours):
        contour = contour.copy()
        contour[:, 0] -= contour[:, 0].min()
        contour[:, 1] -= contour[:, 1].min()
        img = np.zeros(shape=(contour[:, 0].max(), contour[:, 1].max()), dtype=np.uint8)
        contour = np.array(contour)
        contour = np.expand_dims(contour, axis=0)
        # cv2.polylines(img, contour, True, i+1, 5)
        cv2.fillPoly(img, contour, 1, lineType=1)
        pplt.add(img)
    pplt.show()


def draw_contours_shapely(*multi_polygons):
    pplt = PPlot()
    for i, multi_polygon in enumerate(multi_polygons):
        pplt.add(multi_polygon)
    pplt.show()


def ___draw_contours_shapely(*multi_polygons):
    for i, multi_polygon in enumerate(multi_polygons):
        ax = plt.subplot(1, len(multi_polygons), i+1)
        patch = PolygonPatch(multi_polygon, fc=[0,0,1,0.5], ec=[0,0,1,0.5], alpha=0.5, zorder=2)
        ax.add_patch(patch)
        # ax.plot()
        l, u, r, d = multi_polygon.bounds
        ax.set_xticks((l, r))
        ax.set_yticks((u, d))
    plt.axis('equal')
    plt.show()

