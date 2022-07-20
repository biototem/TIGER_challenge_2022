from typing import Tuple

import numpy as np
import cv2

from utils import argmax_numpy, SimpleMultiPolygon, SimplePolygon, Timer


def merge_evaluate(merged: np.ndarray, label: np.ndarray, box: Tuple[int, int, int, int], T: Timer = None):
    """
    旧方法，在预测融合图的基础上进行评估
    """
    merged = merged.copy()
    label = label.copy()
    if box is not None:
        if T: T.track(' -> box matching')
        l, u, r, d = map(int, box)
        label[:, :l, :] = 0
        label[:, r+1:, :] = 0
        label[:u, :, :] = 0
        label[d+1:, :, :] = 0

    if T: T.track(' -> doing argmax')
    mask = label.sum(axis=2).astype(bool)
    argmaxed = argmax_numpy(merged, axis=2)
    argmaxed[~mask, :] = 0
    merged[~mask, :] = 0

    if T: T.track(' -> evaluating')
    lab_value = lab_pixes = label.sum(axis=(0, 1))
    pre_pixes = argmaxed.sum(axis=(0, 1))
    pre_value = merged.sum(axis=(0, 1))

    inter_pixes = (argmaxed * label).sum(axis=(0, 1))
    inter_value = (merged * label).sum(axis=(0, 1))

    dice_pixes = 2.0 * inter_pixes / (lab_pixes + pre_pixes + 1e-17)
    dice_value = 2.0 * inter_value / (lab_value + pre_value + 1e-17)

    result = {
        'lab_value': list(map(float, lab_value)),
        'lab_pixes': list(map(float, lab_pixes)),
        'pre_value': list(map(float, pre_value)),
        'pre_pixes': list(map(float, pre_pixes)),
        'inter_value': list(map(float, inter_value)),
        'inter_pixes': list(map(float, inter_pixes)),
        'dice_value': list(map(float, dice_value)),
        'dice_pixes': list(map(float, dice_pixes)),
    }
    del merged
    del label

    return result


def info_evaluate(info: dict, label: np.ndarray, box: Tuple[int, int, int, int] = None, T: Timer = None):
    """
    新方法的一半（临时措施），用轮廓信息和 label 标签做评估
     -> 原因解释： 由于训练阶段的数据特点， 数据集中的 label 应当以 numpy 存储
     -> 而 eval_test 流程依赖 predict_set 数据集，因此 label 格式无法调整
     -> 至于将来如果构建了 full_image_dataset 数据集，其中大部分区域是没有 label 的
     -> 因而也根本不可能用于 eval_test 流程，这意味着轮廓绘定的 label 除此处之外毫无意义
     -> 故此，损失一些时间效率，规避代码结构上的改动风险，是值得的
    """
    # 第一步，标签轮廓提取
    if T: T.track(' -> finding label contours')
    label_normals, label_atrophics = find_label_contours(label)

    # 第二步，预测轮廓拆分
    if T: T.track(' -> dividing predict contours')
    predict_normals, predict_atrophics = divide_predict_contours(info)

    # 第三步, Box 取齐 (只有 Box 内的轮廓有评估价值)
    if box is not None:
        if T: T.track(' -> box matching')
        l, u, r, d = box
        box = SimplePolygon([(l, u), (l, d), (r, d), (r, u)])
        label_normals = SimpleMultiPolygon.asSimple(label_normals & box)
        label_atrophics = SimpleMultiPolygon.asSimple(label_atrophics & box)
        predict_normals = SimpleMultiPolygon.asSimple(predict_normals & box)
        predict_atrophics = SimpleMultiPolygon.asSimple(predict_atrophics & box)

    # 第四步，分类型评估
    if T: T.track(' -> evaluating normals')
    normal_evaluator = single_evaluate(predict_normals, label_normals, T=T.tab())
    if T: T.track(' -> evaluating atrophics')
    atrophic_evaluator = single_evaluate(predict_atrophics, label_atrophics, T=T.tab())
    return {
        'normal': normal_evaluator,
        'atrophic': atrophic_evaluator,
    }


def find_label_contours(label: np.ndarray):

    normals, _ = cv2.findContours(
        image=label[:, :, 2].astype(np.uint8),
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_SIMPLE
    )
    atrophics, _ = cv2.findContours(
        image=label[:, :, 3].astype(np.uint8),
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_SIMPLE
    )
    # cv2 格式, List[contour] -> contour(n, 1, point(y, x))
    # shapely 格式 List[contour] -> contour(n, point(y, x))
    # Shapely 对象 (经过包装的) -> 本项目后续全部参数传递均使用该包装对象
    normals = SimpleMultiPolygon(list(c[:, 0, :] for c in normals))
    atrophics = SimpleMultiPolygon(list(c[:, 0, :] for c in atrophics))
    normals = normals.smooth(3)
    atrophics = atrophics.smooth(3)
    normals = normals.simplify(0.5)
    atrophics = atrophics.simplify(0.5)
    return SimpleMultiPolygon.asSimple(normals), SimpleMultiPolygon.asSimple(atrophics)


def divide_predict_contours(info: dict):
    normals = []
    atrophics = []
    for contour in info['contours']:
        if contour['base']['tp'] == 'normal':
            normals.append(contour['contour'])
        else:
            atrophics.append(contour['contour'])
    normals = SimpleMultiPolygon(normals)
    atrophics = SimpleMultiPolygon(atrophics)
    return normals, atrophics


def single_evaluate(predict: SimpleMultiPolygon, label: SimpleMultiPolygon, T: Timer = None):
    # 面积评估
    if T: T.track(' -> evaluating area')
    area_pre = predict.area
    area_lab = label.area
    area_inter = (label & predict).area
    area_union = (label | predict).area
    dice = 2 * area_inter / (area_pre + area_lab)

    results = {
        'area': {
            'pre': area_pre,
            'lab': area_lab,
            'inter': area_inter,
            'union': area_union,
            'dice': dice,
        }
    }

    # 轮廓评估
    num_pre = len(predict)
    num_lab = len(label)
    for threshold in [0.3, 0.5, 0.8]:
        if T: T.track(f' -> evaluating contour matches {threshold}')
        pairs = contour_match(predict, label, threshold=threshold)
        num_TP = len(pairs)
        num_FP = num_pre - num_TP
        num_FN = num_lab - num_TP
        score_pre = num_TP / (num_pre + 1e-17)
        score_rec = num_TP / (num_lab + 1e-17)
        score_f05 = fscore(0.5, score_pre, score_rec)
        score_f1 = fscore(1, score_pre, score_rec)
        score_f2 = fscore(2, score_pre, score_rec)
        results[str(threshold)] = {
            'TP': num_TP,
            'FP': num_FP,
            'FN': num_FN,
            'pre': score_pre,
            'rec': score_rec,
            'f05': score_f05,
            'f1': score_f1,
            'f2': score_f2,
        }

    return results


def fscore(beta: float, pre: float, rec: float) -> float:
    pr = 1 / (1 + beta**2)
    pp = 1 - pr
    return pre * rec / (pp*pre + pr*rec + 1e-17)


def contour_match(predict: SimpleMultiPolygon, label: SimpleMultiPolygon, threshold: float = 0.5):
    pres = list(predict)
    labs = list(label)
    n = len(pres)
    m = len(labs)
    # 匹配矩阵
    matrix = np.empty(shape=(n, m), dtype=np.float32)
    for i in range(n):
        for j in range(m):
            inter = pres[i] & labs[j]
            union = pres[i] | labs[j]
            matrix[i, j] = inter.area / (union.area + 1e-17)
    # 权值排序
    keys = [(i, j) for i in range(n) for j in range(m)]
    keys.sort(key=lambda p: matrix[p[0], p[1]], reverse=True)
    # 优先列
    pairs = []
    k1s = set()
    k2s = set()
    for k1, k2 in keys:
        if matrix[k1, k2] < threshold: break
        if k1 in k1s or k2 in k2s: continue
        pairs.append((pres[k1], labs[k2]))
        k1s.add(k1)
        k2s.add(k2)
    return pairs
