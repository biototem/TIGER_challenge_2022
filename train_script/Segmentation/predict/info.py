import sys
import math
import numpy as np
from PIL import ImageDraw, Image

from utils import PPlot, SimpleMultiPolygon, argmax_numpy, SimplePolygon


def create_info(contours: SimpleMultiPolygon, merged: np.ndarray, show: bool = False):
    # merged_visual = Canvas(1).with_label(merged, hyp['visual.color.fill']).image
    # temp = np.zeros(shape=merged.shape[:2], dtype=np.uint8)

    info_contours = []
    for contour in contours:
        if not contour:
            print('----------------- empty contour found -----------------------', file=sys.stderr)
            print(contours.is_valid())
            print(contour.is_valid())
            print(contour)
            print('----------------- empty contour found -----------------------', file=sys.stderr)
            continue
        # 获得形状大小
        l, u, r, d = contour.bounds
        l = max(0, math.floor(l))
        u = max(0, math.floor(u))
        r = min(merged.shape[1] - 1, math.ceil(r))
        d = min(merged.shape[0] - 1, math.ceil(d))
        # l, u = math.floor(l), math.floor(u)
        # r, d = math.ceil(r), math.ceil(d)
        # 平移形状, 准备绘图
        polygon = contour - (l, u)
        # 采用 PIL 绘图方法, 有两个原因:
        # 1. 形状偏小, cv2 和 PIL 速度差距可以接受
        # 2. 同样因为形状偏小, 由边界舍入引起的 零一误差 较大
        #    PIL 可将边界绘制为 0.5, 从原理上规避 零一误差
        #    (现在: canvas.sum() - polygon.area == 1)
        #    (这个误差是 零一误差沿闭合边界积分的"留数", 固定为 1)
        # 之前是上面那样没错，但现在做了轮廓简化后，边界误差严重增大
        # 后续再想办法解决，现在直接边界像素一律纳入统计算了
        # canvas = Image.new(mode="L", size=(r - l + 1, d - u + 1))
        # dw = ImageDraw.Draw(canvas)
        # dw.polygon(list(polygon), fill=2, outline=1)
        # canvas = np.asarray(canvas)
        # canvas = canvas / 2
        canvas = Image.new(mode="L", size=(r - l + 1, d - u + 1))
        dw = ImageDraw.Draw(canvas)
        # 这里, 只画外轮廓
        dw.polygon(polygon.sep_p(), fill=1, outline=1)
        canvas = np.asarray(canvas)

        # 这里, 如若采用 cv2 进行绘图, 边界误差得不到抑制, 则区域信息的提取必然出错
        # canvas = np.zeros(shape=(d-u+1, r-l+1, 1), dtype=np.uint8)
        # cv_draw = np.array(list(polygon), dtype=np.int32).reshape(1, -1, 2)
        # cv2.fillPoly(canvas, cv_draw, color=1, lineType=cv2.LINE_8)
        info = make_merge_info(polygon, merged[u:d + 1, l:r + 1, :], mask=canvas)
        # 滤除无效轮廓
        if not filter_info(info): continue
        info['contour'] = list(contour.sep_p())
        info['bounds'] = list(contour.bounds)
        info['center'] = list(contour.center)
        info_contours.append(info)

        if show:
            info_image = canvas * merged[u:d + 1, l:r + 1, :]
            PPlot().add(
                canvas,
                merged[u:d + 1, l:r + 1, (1, 2, 3)],
                polygon.scale(1, -1) + (0, d - u),
                info_image[:, :, (1, 2, 3)]
            ).show()

    # 全局信息汇总
    normal_contour_number = 0
    atrophic_contour_number = 0
    normal_contour_pixes = 0
    atrophic_contour_pixes = 0
    normal_predict_pixes = 0
    atrophic_predict_pixes = 0
    for ic in info_contours:
        if ic['base']['tp'] == 'normal':
            normal_contour_number += 1
            normal_contour_pixes += ic['base']['pixes']
        else:
            atrophic_contour_number += 1
            atrophic_contour_pixes += ic['base']['pixes']
        normal_predict_pixes += ic['base']['nm']
        atrophic_predict_pixes += ic['base']['at']

    return {
        'size': list(merged.shape[:2]),
        'normal_contour_number': normal_contour_number,
        'normal_contour_pixes': normal_contour_pixes,
        'normal_predict_pixes': normal_predict_pixes,
        'atrophic_contour_number': atrophic_contour_number,
        'atrophic_contour_pixes': atrophic_contour_pixes,
        'atrophic_predict_pixes': atrophic_predict_pixes,
        'contours': info_contours
    }


def filter_info(info):
    # 轮廓面积不小于定值
    if info['base']['pixes'] < 100: return False
    return True


def make_merge_info(polygon: SimplePolygon, hotmap: np.ndarray, mask: np.ndarray):
    """
    统计轮廓的各种信息特征
    主要包括:
    一, 简单特征:
        1. number 像素数
    二, 统计学特征:
        1. 0阶矩 -> 预测总值
        2. 1阶矩 -> 预测均值
        3. 2阶矩 -> 预测方差
        4. 3阶矩 -> 预测偏度
        5. 置信度 / 置信区间
    三, 形态学特征:
        1. 面积 -> 总像素数 - 1 (忽略)
        2. 周长
        3. 周径比
        4. 宽长比
        5. 圆形度
        6. 凸形度
    用途:
        argmax 像素数 用于确定预测轮廓的主要类型 (正常 / 异常)
        统计学特征备用 (展示或检验时)
        形态学特征用于评估轮廓的准确性 (滤除无效轮廓)
    """
    # 预测值 - 边界像素按 1 统计
    hotpoints = hotmap[mask.astype(bool)]
    bgv = hotpoints[:, 0]
    bdv = hotpoints[:, 1]
    nmv = hotpoints[:, 2]
    atv = hotpoints[:, 3]

    # 统计学信息 - 边界像素按 1 统计
    azs = [(0.90, 1.64), (0.95, 1.96), (0.99, 2.58)]
    # 1.不同置信度下 轮廓为背景 的 置信区间
    bgw = [statistics_level(bgv, az) for az in azs]
    # 2.不同置信度下 轮廓为正常 的 置信区间
    nmw = [statistics_level(nmv, az) for az in azs]
    # 3.不同置信度下 轮廓为异常 的 置信区间
    atw = [statistics_level(atv, az) for az in azs]

    # 形态学信息
    #     1. 面积 -> 总像素数 - 1 (忽略)
    #     2. 周长
    #     3. 周径比
    #     4. 宽长比
    #     5. 圆形度
    #     6. 凸度（面积）
    #     7. 凸值（周长）
    bounds = polygon.bounds
    area = polygon.area
    perimeter = polygon.perimeter
    # mmr -> 最小外接矩形
    mrr = polygon.mini_rect
    w, h = get_xy_from_rect(mrr)
    wph = w / (h + 1e-17)
    ppw = perimeter / (w + 1e-17)
    circularity = 4 * np.pi * area / perimeter ** 2
    # ch -> 凸包络
    ch = polygon.convex
    convexity_s = area / (ch.area + 1e-17)
    convexity_l = ch.perimeter / (perimeter + 1e-17)

    # 基本信息 - 边界像素按 1 统计
    points = argmax_numpy(hotpoints, axis=1)
    pixes = mask.sum()
    bgs = points[:, 0].sum()
    bds = points[:, 1].sum()
    nms = points[:, 2].sum()
    ats = points[:, 3].sum()
    # 类型 -> 根据预测像素数统计，正常/异常 比例大于阈值 (0.7:0.3) 时认定为正常
    tp = 'normal' if 0.3 * nms > 0.7 * ats else 'atrophic'
    # 轮廓分数 -> 分数越高，形态学意义上越完整，越可能是一个独立轮廓（但现在的评价方式似乎并不合理）
    ct_score = 2 * abs(convexity_s * convexity_l) ** 0.5
    # 肾小管分数 -> 分数越高，越可能是真正的肾小管，而非虚假预测（采用 0.95 置信度下背景置信区间大于0.5的长度占比）
    rn_score = 1 - max(0, bgw[1][3] - 0.5) / max(0.5 - bgw[1][2], bgw[1][3] - 0.5)
    # 类型分数 -> 分数越高，在 正常/异常 的分类上越不容易出错（采用 0.95 置信度下目标置信区间低于0.5的长度占比）
    if tp == 'normal':
        tt, wt = nmw, atw
    else:
        tt, wt = atw, nmw
    tp_score = 1 - max(0, wt[1][3] - tt[1][2]) / max(tt[1][3] - wt[1][2], wt[1][3] - tt[1][2])
    return {
        'contour': None,
        'bounds': None,
        'center': None,
        'base': {
            'pixes': float(pixes),
            'tp': tp,
            'nm': float(nms),
            'at': float(ats),
            'ct_score': float(ct_score),
            'rn_score': float(rn_score),
            'tp_score': float(tp_score),
        },
        'shape': {
            'area': area,
            'length': perimeter,
            'wph': wph,
            'ppw': ppw,
            'circularity': circularity,
            'convexity_s': convexity_s,
            'convexity_l': convexity_l,
        },
        'statistics': {
            'bg': {
                'pixes': float(bgs),
                'mass': float(bgv.sum()),
                'mean': float(bgv.sum() / pixes),
                'val': float((bgv ** 2).sum() / pixes),
                'r3': float((bgv ** 3).sum() / pixes),
                'deviation': bgw,
            },
            'nm': {
                'pixes': float(nms),
                'mass': float(nmv.sum()),
                'mean': float(nmv.sum() / pixes),
                'val': float((nmv ** 2).sum() / pixes),
                'r3': float((nmv ** 3).sum() / pixes),
                'deviation': nmw,
            },
            'at': {
                'pixes': float(ats),
                'mass': float(atv.sum()),
                'mean': float(atv.sum() / pixes),
                'val': float((atv ** 2).sum() / pixes),
                'r3': float((atv ** 3).sum() / pixes),
                'deviation': atw,
            },
        },
    }


def statistics_level(src: np.ndarray, level_z=(0.95, 1.96)):
    mean, std = src.mean(), src.std()
    a, z = level_z
    r = z * std / (len(src) ** 0.5 + 1e-17)
    return float(a), float(r), float(mean - r), float(mean + r)


def get_xy_from_rect(rect: SimplePolygon):
    """
    根据周长,面积信息逆向计算出矩形边长 a,b
    """
    s = rect.area
    l = rect.perimeter
    delta = (l * l / 4 - 4 * s) ** 0.5
    a = l / 4 + delta / 2
    b = l / 4 - delta / 2
    return a, b

# class StatisticsLevel(object):
#     def __init__(self, levels = (0.95, )):
#         self.levels = levels
#
#     def

# # 背景均值和标准差
# bge, bgq = bgv.mean(), bgv.std()
# # 置信度水平及对应 Z 值
# zs = [(0.90, 1.64), (0.95, 1.96), (0.99, 2.58)]
# # 置信度水平及对应半径
# rs = [(a, z * bgq / bgs**0.5) for a, z in zs]
# # 置信度水平,半径,置信区间
# ws = [(a, r, bge-r, bge+r) for a, r in rs]
# # 2.轮廓为异常 的 置信区间
