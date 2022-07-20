import json
from typing import Dict

import cv2
import numpy as np

from basic import join
from utils import Shape, SimplePolygon, ComplexPolygon, ComplexMultiPolygon, Region


# from .label_map import mapper
# sorted_items = list(sorted(mapper.items(), key=lambda x: x[1]['priority']))
# class_names = [k for k, _ in sorted_items]
# class_value = [v['label'] for _, v in sorted_items]
# del sorted_items
mapper = {
    'MASK': 0,
    'ROI': 2,
    'STROMA': 2,
    'NORMAL': 3,
    'OTHER': 3,
    'TUMOR': 4,
    'NECROSIS': 5,
    'VESSEL': 6,
    'IMMUNE CELLS': 7,
}


class ShapeLoader(object):
    """
    For label.
    """
    def __init__(self, path: str, zoom: float, **kwargs):
        self.labels = load(path, zoom)

    def __call__(self, grid: dict):
        # Numpy-array to return
        # Initializing value 1 means all pixes perform like 'background' in white or black
        # If there are any organizations, class['mask'] will set them to 0 which means 'undefined' at first
        # Then class['roi'] will set organizations in box to 2 which defined as 'stroma'
        # At last, other contours are drawn following their area
        temp = np.ones(shape=(grid['h'], grid['w']), dtype=np.uint8)
        for value, shapes in self.labels:
            # Make a copy of shape and normalize the
            shapes = shapes.copy()
            # Move the Shape into Point-O for scaling and rotating
            shapes -= (grid['x'], grid['y'])
            # For scaling: window size up -> shape size down
            # For rotate: window anti-clock -> shape clockwise
            shapes /= grid['scaling']
            shapes **= grid['degree']
            # Move [-A, A] to [0, 2A] for drawing contours
            shapes += (grid['w'] / 2, grid['h'] / 2)
            # Drop contours out of region -> to prevent cv2 exception
            shapes &= Region(left=0, up=0, right=grid['w'], down=grid['h'])
            # 数据集接收的 areas 全部都是 ComplexMultiPolygon, 对其进行分解作业
            for shape in shapes.sep_out():
                outers, inners = shape.sep_p()
                contours = [np.array(contour, dtype=int) for contour in (outers, *inners) if bool(contour)]
                cv2.fillPoly(temp, contours, value)
        return temp


def load(path: str, zoom: float) -> Dict[str, Shape]:
    # print(json.dumps(area, default=sep))
    # 加载组织轮廓存储信息
    with open(join(path), 'r') as f:
        area_json = json.load(f)
    area = {}
    for key, shape_json in area_json.items():
        key = key.upper()
        if shape_json['type'].upper() == 'REGION':
            outer = shape_json['points']
            area[key] = SimplePolygon(outer=outer).simplify()
        elif shape_json['type'].upper() == 'COMPLEXPOLYGON':
            outer, inners = shape_json['points']
            area[key] = ComplexPolygon(outer, *inners).simplify()
        elif shape_json['type'].upper() == 'COMPLEXMULTIPOLYGON':
            outers, inners, adjacencies = shape_json['points']
            area[key] = ComplexMultiPolygon(outers=outers, inners=inners, adjacencies=adjacencies).simplify()
        else:
            raise TypeError(f'unrecognized type: {shape_json["type"]}')
        assert area[key] != Shape.EMPTY, f'出现非法图形! 请检查: {path}: {key}'
        area[key] /= zoom
    assert ('MASK' in area) and ('ROI' in area), '组织轮廓标签错误! 本项目要求组织轮廓必须包含 MASK 和 ROI!'

    # 按轮廓大小排序 -> ROI 和 MASK 除外
    # 任何标签轮廓都必须与组织轮廓相交, 否则说明组织轮廓分割错误
    mask = area.pop('MASK')
    roi = area.pop('ROI') & mask

    labels = []
    for key, shapes in area.items():
        assert shapes.is_joint(mask)
        shapes &= mask
        val = mapper[key]
        labels.extend((val, shape) for shape in shapes)
    labels = sorted(labels, key=lambda pair: pair[1].area, reverse=True)
    labels = [(0, mask), (2, roi)] + labels
    return labels
