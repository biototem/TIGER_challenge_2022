import cv2
import os
import numpy as np
from typing import Iterable, Union
from lib.list_tool import list_multi_get_with_ids, list_multi_get_with_bool, list_multi_set_with_ids, list_multi_set_with_bool
from lib import contour_tool
from lib.point_tool import get_shortest_link_pair
from skimage.draw import disk as sk_disk


def softmax(x, axis=-1):
    return np.exp(x)/ np.sum(np.exp(x), axis=axis, keepdims=True)


def get_pg_id(epoch, process_control):
    # if epoch < process_control[0]:
    #     pg_id = 1
    # elif process_control[0] <= epoch < process_control[1]:
    #     pg_id = 2
    # elif process_control[1] <= epoch:
    #     pg_id = 3
    # else:
    #     raise AssertionError()
    pg_id = 1
    for i in process_control:
        if epoch >= i:
            pg_id += 1
        else:
            break
    return pg_id


def get_pg_name(ori_name, cur_epoch, process_control):
    pg_id = get_pg_id(cur_epoch, process_control)
    base, ext = os.path.splitext(ori_name)
    out_name = base + '_p{}'.format(pg_id) + ext
    return out_name


def patch_merge_func(self, patch_result, patch_result_new, patch_mask):
    '''
    默认的滑窗图合并函数，合并时取最大值
    :param self:                引用大图块自身，用于实现某些特殊用途，一般不使用
    :param patch_result:        当前滑窗区域的结果
    :param patch_result_new:    新的滑窗区域的结果
    :param patch_mask:          当前掩码，用于特殊用途，这里不使用
    :return: 返回合并后结果和更新的掩码
    '''
    new_result = patch_result + patch_result_new
    new_mask = patch_mask + 1
    # new_result = np.maximum(patch_result, patch_result_new)
    # new_mask = np.ones_like(patch_mask)
    return new_result, new_mask


def pm_post_process(bin_im):
    '''
    后处理，目前是先膨胀1次，再腐蚀1次。图像后处理，目的，区域链接起来，然后将修整一下毛边
    :param bin_im:
    :return:
    '''
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    y = bin_im
    y = cv2.dilate(y, k, iterations=1)
    y = cv2.erode(y, k, iterations=1)
    if y.ndim == 2:
        y = y[:, :, None]
    return y


def get_pts_from_hm(hm: np.ndarray, prob: float=0.5):
    '''
    从热图中生成中心点
    :param hm: 就是模型输出的预测热图  128x128x1
    :param prob:
    :return:
    '''
    assert hm.ndim == 3
    assert hm.shape[2] == 1
    hm_bin = (hm > prob).astype(np.uint8)
    contours = contour_tool.find_contours(hm_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = np.array([contour_tool.make_bbox_from_contour(c) for c in contours], np.float32)
    if len(boxes) > 0:
        pts = (boxes[:, :2] + boxes[:, 2:]) / 2 # 求这个外接矩阵的盒子的中心
        pts = np.round(pts).astype(np.int)
        pts = list(pts)
    else:
        pts = []
    return pts



def get_pts_from_hm_with_probs(hm: np.ndarray, prob: float=0.5):
    '''
    从热图中生成中心点
    :param hm: 就是模型输出的预测热图  128x128x1
    :param prob:
    :return:
    '''
    assert hm.ndim == 3
    assert hm.shape[2] == 1
    hm_bin = (hm > prob).astype(np.uint8)
    contours = contour_tool.find_contours(hm_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = np.array([contour_tool.make_bbox_from_contour(c) for c in contours], np.float32)

    if len(boxes) > 0:
        pts = (boxes[:, :2] + boxes[:, 2:]) / 2 # 求这个外接矩阵的盒子的中心
        pts = np.round(pts).astype(np.int)
        pts = list(pts)
        pts = [list(item) for item in pts]
        pix_result = [float(hm[x,y]) for (x,y) in pts]
    else:
        pts = []
        pix_result = []
    return pts,pix_result


def get_cls_pts_from_hm_xiefeng(det_pts, cls_hm: np.ndarray):
    '''
    从热图中生成中心点
    :param hm:
    :param prob:
    :return:
    '''
    assert cls_hm.ndim == 3
    assert cls_hm.shape[2] >= 1
    pts_cls = np.zeros([len(det_pts)], np.int)
    pts_cls_rate = np.zeros([len(det_pts)], np.float32)   # 这里是我加的
    hw = cls_hm.shape[:2]
    pred_cls_rate_all = []
    for i, pt in enumerate(det_pts):
        probs = np.zeros([cls_hm.shape[2]], np.float32)
        rr, cc = sk_disk(pt, radius=3, shape=hw)
        for c in range(cls_hm.shape[2]):
            probs[c] = cls_hm[rr, cc, c].sum()
        cls = np.argmax(probs)
        pts_cls[i] = cls
        pts_cls_rate[i] = probs[cls]   # 把概率值最大的保存下来，就代表了是这个细胞核的分类结果
        pred_cls_rate_all.append(probs)
    return pts_cls,pts_cls_rate,pred_cls_rate_all

def get_cls_pts_from_hm(det_pts, cls_hm: np.ndarray):  # 文泰原来的代码
    '''
    从热图中生成中心点
    :param hm:
    :param prob:
    :return:
    '''
    assert cls_hm.ndim == 3
    assert cls_hm.shape[2] >= 1
    pts_cls = np.zeros([len(det_pts)], np.int)
    hw = cls_hm.shape[:2]
    for i, pt in enumerate(det_pts):
        probs = np.zeros([cls_hm.shape[2]], np.float32)
        rr, cc = sk_disk(pt, radius=3, shape=hw)
        for c in range(cls_hm.shape[2]):
            probs[c] = cls_hm[rr, cc, c].sum()
        cls = np.argmax(probs)
        pts_cls[i] = cls
    return pts_cls

def get_cls_pts_from_hm_2(det_pts, cls_hm: np.ndarray):
    '''
    从热图中生成中心点，加入返回概率的功能
    :param hm:
    :param prob:
    :return:
    '''
    assert cls_hm.ndim == 3
    assert cls_hm.shape[2] >= 1
    # pts_cls = np.zeros([len(det_pts)], np.int)
    pts_probs = np.zeros([len(det_pts), cls_hm.shape[2]], np.float32)
    hw = cls_hm.shape[:2]
    for i, pt in enumerate(det_pts):
        probs = np.zeros([cls_hm.shape[2]], np.float32)
        rr, cc = sk_disk(pt, radius=3, shape=hw)
        for c in range(cls_hm.shape[2]):
            probs[c] = cls_hm[rr, cc, c].sum()
        probs = probs / np.sum(probs)
        pts_probs[i] = probs
    return pts_probs


def find_too_far_pts(wait_check_pts: np.ndarray, close_pts, close_dist: float=6):
    '''
    删除距离指定顶点过近的顶点
    :param wait_check_pts:
    :param close_pts:
    :param close_dist:
    :return:
    '''
    if len(wait_check_pts) == 0:
        return wait_check_pts
    wait_check_pts = np.asarray(wait_check_pts)
    close_pts = np.asarray(close_pts)
    keep_b = np.ones([len(wait_check_pts)], np.bool)
    for pt in close_pts:
        dists = np.linalg.norm(wait_check_pts - pt[None,], 2, axis=1)
        keep_b[dists <= close_dist] = False
    if np.any(keep_b):
        keep_ids = np.argwhere(keep_b)[:, 0]
    else:
        keep_ids = []
    return keep_ids


def draw_hm_circle(ori_im: np.ndarray, pred_pts: np.ndarray, label_pts: np.ndarray, close_dist=6):
    '''
    绘制检测图。
    预测点绘制圆环，检测到为黄，否则为红
    标签点绘制原点，检测到为黄，否则为绿
    :param ori_im:
    :param pred_pts:
    :param label_pts:
    :param close_dist:
    :param radius:
    :return:
    '''
    # assert isinstance(pred_pts, np.ndarray)
    # assert isinstance(label_pts, np.ndarray)
    pred_b = np.zeros(len(pred_pts), dtype=np.bool)
    label_b = np.zeros(len(label_pts), dtype=np.bool)
    for p_id, p_pt in enumerate(pred_pts):
        if len(label_pts) != 0:
            dists = np.linalg.norm(p_pt[None,] - label_pts, axis=1) # 预测的中心点和 label中心点之间的距离
            close_bools = dists <= close_dist
            np.logical_or(label_b, close_bools, out=label_b)
            if np.any(close_bools):
                pred_b[p_id] = True

    pred_fakefound_color = (255, 0, 0)
    pred_found_color = (255, 255, 0)
    label_nofound_color = (0, 255, 0)
    label_found_color = (255, 255, 0)

    draw_im = np.zeros([ori_im.shape[0], ori_im.shape[1], 3], np.uint8)

    for b, pt in zip(pred_b, pred_pts):
        if b:
            color = pred_found_color
        else:
            color = pred_fakefound_color
        # draw_im = cv2.circle(cv2.UMat(draw_im), tuple(pt)[::-1], close_dist, color, 1).get()
        draw_im = cv2.circle(draw_im, tuple(pt[::-1]), close_dist, color, 1)

    for b, pt in zip(label_b, label_pts):
        if b:
            color = label_found_color
        else:
            color = label_nofound_color
        # draw_im = cv2.circle(cv2.UMat(draw_im), tuple(pt)[::-1], 3, color, -1).get()
        draw_im = cv2.circle(draw_im, tuple(pt[::-1]), 3, color, -1)

    mix_pic = np.where(np.any(draw_im > 0, -1, keepdims=True), draw_im, ori_im)

    return mix_pic


def calc_cls_with_con_and_zone(con, cm, cls_list):
    '''
    使用轮廓提取出概率热图中的区域，分类概率总和，总和最大的类别将视为该轮廓的类别
    '''
    max_cls_precent = -1.
    max_cls_id = -1

    bbox = contour_tool.make_bbox_from_contour(con)
    cm = cm[bbox[0]: bbox[2], bbox[1]: bbox[3]]
    zone_bm = np.zeros([cm.shape[0], cm.shape[1], 1], np.uint8)
    con = con - bbox[:2][::-1][None, None]

    zone_bm = cv2.drawContours(zone_bm, [con], -1, 1, -1)
    select_cm = cm * zone_bm
    for cls in cls_list:
        _cls_p = select_cm[..., cls].sum()
        if _cls_p > max_cls_precent:
            max_cls_precent = _cls_p
            max_cls_id = cls

    return max_cls_precent, max_cls_id


def class_map_to_contours(cm, cls_list, cls_score_thresh, use_post_pro):
    '''
    转换输入的分类图到轮廓。
    :param cm: 输入的类别图
    :param cls_list: 需要生成轮廓的类别列表
    :param cls_score_thresh: 类别阈值
    :param use_post_pro: 是否对类别图使用后处理
    :param func: 选择使用哪个函数
    :return:
    '''
    assert cm.ndim == 3
    assert len(cls_list) > 0

    # TO CHECK
    has_class_bm = np.any(cm[..., cls_list] > cls_score_thresh, -1, keepdims=True).astype(np.uint8)

    if use_post_pro:
        has_class_bm = pm_post_process(has_class_bm)

    # cv2.RETR_EXTERNAL 代表只查找外轮廓
    all_contours, _ = cv2.findContours(has_class_bm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_contours = contour_tool.tr_cv_to_my_contours(all_contours)
    # 检查轮廓，如果轮廓顶点少于3，则抛弃
    all_contours = [c for c in all_contours if len(c) >= 3]

    out_contour_list = []
    out_class_list = []

    for con in all_contours:
        max_cls_precent, max_cls_id = calc_cls_with_con_and_zone(con, cm, cls_list)

        out_contour_list.append(con)
        out_class_list.append(max_cls_id)

    return out_contour_list, out_class_list


def class_pm_to_bm(pm, cls_p):
    bm = (pm > np.reshape(cls_p, [1, 1, len(cls_p)])).astype(np.uint8)
    return bm


def class_map_to_contours2(pm, pred_cls_list, cls_prob, need_merge_cls_list, use_post_pro):
    '''
    转换输入的分类图到轮廓。
    :param pm: 输入的概率图
    :param pred_cls_list: 需要生成轮廓的类别列表
    :param cls_prob: 需要生成轮廓的类别列表
    :param need_merge_cls_list: 需要合并的类别列表
    :param use_post_pro: 是否对类别图使用后处理
    :return:
    '''
    assert pm.ndim == 3
    assert len(pred_cls_list) > 0
    assert len(cls_prob) == len(pred_cls_list)
    assert set(need_merge_cls_list).issubset(pred_cls_list)

    # 后处理
    if use_post_pro:
        pm = pm_post_process(pm)

    # 找出每一类的轮廓
    bm = class_pm_to_bm(pm, cls_prob)

    out_contour_list = []
    out_class_list = []

    for cls in pred_cls_list:
        contours = contour_tool.find_contours(bm[:, :, cls], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out_contour_list.extend(contours)
        out_class_list.extend([cls] * len(contours))

    # 合并指定类的轮廓，并保留每个合并组中最大的类别
    wait_merge_contour_list = []
    wait_merge_class_list = []

    dont_merge_contour_list = []
    dont_merge_class_list = []

    for con, cls in zip(out_contour_list, out_class_list):
        if cls in need_merge_cls_list:
            wait_merge_contour_list.append(con)
            wait_merge_class_list.append(cls)
        else:
            dont_merge_contour_list.append(con)
            dont_merge_class_list.append(cls)

    if len(wait_merge_contour_list) > 0:
        merge_contour_list, max_contour_ids = contour_tool.merge_multi_contours_sort_by_area(wait_merge_contour_list)
        merge_class_list = list_multi_get_with_ids(wait_merge_class_list, max_contour_ids)
    else:
        merge_contour_list = []
        merge_class_list = []

    final_contour_list = []
    final_class_list = []

    final_contour_list.extend(merge_contour_list)
    final_contour_list.extend(dont_merge_contour_list)

    final_class_list.extend(merge_class_list)
    final_class_list.extend(dont_merge_class_list)

    return final_contour_list, final_class_list


def calc_a_sample_info(out_cm, label_cm, cls_list, cls_score_thresh: Union[float, Iterable], match_iou_thresh=0.3, use_post_pro=False, ignore_board=0., *, direct_input=False, func=1):
    '''
    :param out_cm:          预测类别图，当direct_input被设置时为轮廓
    :param label_cm:        标签类别图，当direct_input被设置时为轮廓
    :param cls_list:        类别列表
    :param cls_score_thresh:    当类别分数大于该值时，认为存在该类，可以是标量或列表
    :param match_iou_thresh:    当IOU大于该值指定的将视为已发现轮廓
    :param use_post_pro:    是否使用后处理
    :param ignore_board:    是否忽略边界
    :param direct_input:    out_cm 和 label_cm 输入是轮廓，不是分类图
    :param func:            遗留函数，待删除
    :return:
    '''
    if direct_input:
        out_contours, out_clss = out_cm
        label_contours, label_clss = label_cm
    else:
        out_contours, out_clss = class_map_to_contours(out_cm, cls_list, cls_score_thresh, use_post_pro)
        label_contours, label_clss = class_map_to_contours(label_cm, cls_list, cls_score_thresh, use_post_pro)

    out_contours_found_count = np.zeros([len(out_contours)], np.int)

    out_clss = np.array(out_clss)
    label_clss = np.array(label_clss)

    d = {
        'fake_found': 0,                                # 假的轮廓，没有标签对应的
        'fake_found_ids_in_out': [],
        'nofound_ids_in_label': [],
    }
    for cls in cls_list:
        d[cls] = {'found_with_true_cls': 0,             # 找到了，包含iou是最大和不是最大的情况
                  'found_with_error_cls': 0,            # 找到了，但却分类错误
                  'found_with_true_with_max_iou': 0,    # 找到了，最高iou的轮廓分类正确
                  'found_with_true_without_max_iou': 0, # 找到了，不是最高iou的轮廓分类正确
                  'nofound': 0,                         # 没找到
                  }

    # 对标签轮廓循环可以更简单
    for i, (lc, lclss) in enumerate(zip(label_contours, label_clss)):
        ious = contour_tool.calc_iou_with_contours_1toN(lc, out_contours)
        pass_out_ids = np.argwhere(ious >= match_iou_thresh)[:, 0].astype(np.int)
        pass_out_ious = ious[pass_out_ids]
        pass_out_clss = out_clss[pass_out_ids]

        if len(pass_out_ids) > 0:
            out_contours_found_count[pass_out_ids] += 1
            sort_out_ids = np.argsort(pass_out_ious)[::-1]
            sort_out_clss = pass_out_clss[sort_out_ids]
            if lclss in sort_out_clss:
                d[lclss]['found_with_true_cls'] += 1
                if sort_out_clss[0] == lclss:
                    d[lclss]['found_with_true_with_max_iou'] += 1
                else:
                    d[lclss]['found_with_true_without_max_iou'] += 1
            else:
                d[lclss]['found_with_error_cls'] += 1
        else:
            d[lclss]['nofound'] += 1
            d['nofound_ids_in_label'].append(i)

    fake_found = out_contours_found_count == 0
    d['fake_found'] = int(fake_found.sum())
    d['fake_found_ids_in_out'].extend(np.argwhere(fake_found)[:, 0].astype(np.int))
    return d


def calc_a_sample_info_points_each_class(pred_pm, label_pm, cls_list, match_distance_thresh_list=(5, 7, 9, 11), use_post_pro=False, use_single_pair=False):
    '''
    :param pred_pm:          预测类别图，当direct_input被设置时为轮廓
    :param label_pm:        标签类别图，当direct_input被设置时为轮廓
    :param cls_list:        类别列表
    :param match_distance_thresh_list:    当IOU大于该值指定的将视为已发现轮廓
    :param use_post_pro:    是否使用后处理
    :param ignore_board:    是否忽略边界
    :param direct_input:    pred_contours 和 label_contours 输入是轮廓，不是分类图
    :return:
    '''
    if isinstance(pred_pm, list):
        pred_centers, pred_cls = pred_pm
    else:
        pred_contours, pred_cls = class_map_to_contours2(pred_pm, cls_list, [0.5]*len(cls_list), {}, use_post_pro)   # 获得预测的每个细胞核的轮廓，
        pred_bboxes = np.array([contour_tool.make_bbox_from_contour(c) for c in pred_contours])
        pred_centers = (pred_bboxes[:, :2] + pred_bboxes[:, 2:]) / 2 if len(pred_bboxes) > 0 else []  # 获得预测细胞核的中心，根据外边界矩阵

    if isinstance(label_pm, list):
        label_centers, label_cls = label_pm   # 标签的中心和分类 id
    else:
        label_contours, label_cls = class_map_to_contours2(label_pm, cls_list, [0.5]*len(cls_list), {}, use_post_pro)
        label_bboxes = np.array([contour_tool.make_bbox_from_contour(c) for c in label_contours])
        label_centers = (label_bboxes[:, :2] + label_bboxes[:, 2:]) / 2 if len(label_bboxes) > 0 else []

    score_table = {}

    if len(label_centers) == 0 or len(pred_centers) == 0:
        for cls in cls_list:
            score_table[cls] = {}
            for distance in match_distance_thresh_list:
                score_table[cls][distance] = {}
                score_table[cls][distance]['found_pred'] = 0
                score_table[cls][distance]['fakefound_pred'] = len(pred_centers)
                score_table[cls][distance]['found_label'] = 0
                score_table[cls][distance]['nofound_label'] = len(label_centers)
                score_table[cls][distance]['pred_repeat'] = 0
                score_table[cls][distance]['label_repeat'] = 0

        return score_table

    for cls in cls_list:  # 遍历获取每个类别的中心坐标，无论是预测的还是label
        score_table[cls] = {}
        pred_selected_bools = np.array(pred_cls, np.int) == cls  # [ True  True  True  True  True  True  True  True  True  True  True  True]
        label_selected_bools = np.array(label_cls, np.int) == cls # [ True  True  True  True  True  True  True  True  True]
        selected_pred_centers = list_multi_get_with_bool(pred_centers, pred_selected_bools)
        selected_label_centers = list_multi_get_with_bool(label_centers, label_selected_bools)

        selected_pred_centers = np.array(selected_pred_centers)   # 存储的是预测的中心点（这个类型）
        selected_label_centers = np.array(selected_label_centers) # 存储的标签的中心点（这个类型）

        for distance in match_distance_thresh_list:
            score_table[cls][distance] = {}

            label_found_count = np.zeros(len(selected_label_centers), np.int)  # [0 0 0 0 0 0 0 0 0]
            pred_found_count = np.zeros(len(selected_pred_centers), np.int) # [0 0 0 0 0 0 0 0 0 0 0 0]

            if not use_single_pair:
                for pi, pred_center in enumerate(selected_pred_centers):
                    if len(selected_label_centers) != 0:
                        dists = np.linalg.norm(pred_center[None,] - selected_label_centers, axis=1)
                        close_bools = dists <= distance
                        label_found_count[close_bools] += 1
                        pred_found_count[pi] += np.array(close_bools, np.int).sum()
            else:  # 进行 这个类别的中心坐标的配对
                pred_pt_ids, label_pt_ids, _ = get_shortest_link_pair(selected_pred_centers, selected_label_centers, distance)  # 根据预测的中心和标签的中心，还有距离找出配对的中心点
                for i in pred_pt_ids:
                    pred_found_count[i] = 1   # [0 0 1 1 1 1 1 1 0 1 0 1]
                for i in label_pt_ids:
                    label_found_count[i] = 1  # [1 1 1 1 1 0 1 1 1]

            found_pred = int((pred_found_count > 0).sum())  # 8
            fakefound_pred = int((pred_found_count == 0).sum())  # 4

            found_label = int((label_found_count > 0).sum())  # 8
            nofound_label = int((label_found_count == 0).sum())  # 1

            pred_repeat = int((pred_found_count > 1).sum())
            label_repeat = int((label_found_count > 1).sum())

            score_table[cls][distance]['found_pred'] = found_pred
            score_table[cls][distance]['fakefound_pred'] = fakefound_pred
            score_table[cls][distance]['found_label'] = found_label
            score_table[cls][distance]['nofound_label'] = nofound_label
            score_table[cls][distance]['pred_repeat'] = pred_repeat
            score_table[cls][distance]['label_repeat'] = label_repeat

    return score_table


def get_contours_center(contours):
    centers = []
    for c in contours:
        centers.append(np.mean(c, 0))
    return centers


def calc_metric(pred_im, label_pos):
    '''

    :param pred_im: [H, W, C]
    :param label_pos: [N, p_yx]
    :return:
    '''
    assert pred_im.ndim == 3
    # assert label_im.ndim == 3
    contours, classes = class_map_to_contours(pred_im, [0], 0.8, use_post_pro=False)
    centers = get_contours_center(contours)
    pred_pos = centers
    label_found_count = np.zeros(len(label_pos), np.int)
    pred_found_count = np.zeros(len(pred_pos), np.int)
    for p_i, p_pos in enumerate(pred_pos):
        p_pos = np.reshape(p_pos, [1, 2])
        p_len = np.sqrt(np.sum(np.square(p_pos - label_pos), 1))
        t = p_len <= 6
        label_found_count[t] += 1
        pred_found_count[p_i] += t.sum(dtype=np.int32)

    label_num = len(label_found_count)
    pred_num = len(pred_found_count)
    label_found_num = (label_found_count > 0).sum()
    label_nofound_num = (label_found_count == 0).sum()
    pred_fake_found_num = (pred_found_count == 0).sum()

    recall = label_found_num / (label_num + 1e-8)
    prec = (pred_num - pred_fake_found_num) / (pred_num + 1e-8)
    f1 = 2*(prec*recall)/(prec+recall+1e-8)

    d = {
        'label_found_num': label_found_num,
        'label_nofound_num': label_nofound_num,
        'pred_fake_found_num': pred_fake_found_num,
        'label_num': label_num,
        'pred_num': pred_num,
        'recall': recall,
        'prec': prec,
        'f1': f1,
    }

    return d



if __name__ == '__main__':
    a = [np.array([[0, 0], [0, 100], [100, 0]]), np.array([[100, 100], [100, 0], [0, 0]])+100]
    b = [1, 2]
    oa, ob = contours_post_pro(a, b)
    print(oa, ob)
