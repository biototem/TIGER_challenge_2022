import numpy as np
from typing import Sized
try:
    from affine_matrix_tool import make_scale, make_move, make_rotate, make_shear
except ModuleNotFoundError:
    from .affine_matrix_tool import make_scale, make_move, make_rotate, make_shear


def get_shortest_link_pair(pt_list1, pt_list2, dist_th: float):
    '''
    求两组点之间最短链接对
    :param pt_list1: 点集1
    :param pt_list2: 点集2
    :param dist_th:  最长配对距离，最短距离超过该阈值后不进行配对。
    :return: 返回点对的编号和他们的距离
    '''
    pt_list1 = np.asarray(pt_list1, np.float32).reshape(-1, 2)
    pt_list2 = np.asarray(pt_list2, np.float32).reshape(-1, 2)

    contact_pts1 = []
    contact_pts2 = []
    contact_distance = []

    if 0 == len(pt_list1) or 0 == len(pt_list2):
        return contact_pts1, contact_pts2, contact_distance

    # 先求出所有可以关联的链接，使用numpy的方法，去掉一个循环
    for pt1_id, pt1 in enumerate(pt_list1):
        ds = np.linalg.norm(pt1[None] - pt_list2, 2, axis=1)
        bs = ds <= dist_th
        pt2_ids = np.argwhere(bs).reshape(-1)

        contact_pts1.extend([pt1_id] * len(pt2_ids))
        contact_pts2.extend(pt2_ids)
        contact_distance.extend(ds[pt2_ids])

        # for pt2_id, pt2 in enumerate(pt_list2):
        #     pt2 = np.array(pt2, np.float32)
        #     d = np.linalg.norm(pt1 - pt2, 2)
        #     if d <= dist_th:
        #         contact_pts1.append(pt1_id)
        #         contact_pts2.append(pt2_id)
        #         contact_distance.append(d)

    out_contact_pts1 = []
    out_contact_pts2 = []
    out_contact_distance = []

    # 对距离进行排序，从全局最短的链接开始
    ind = np.argsort(contact_distance)
    for cur_pair_id in ind:
        pt1_id = contact_pts1[cur_pair_id]
        pt2_id = contact_pts2[cur_pair_id]
        if pt1_id not in out_contact_pts1 and pt2_id not in out_contact_pts2:
            out_contact_pts1.append(pt1_id)
            out_contact_pts2.append(pt2_id)
            out_contact_distance.append(contact_distance[cur_pair_id])

    return out_contact_pts1, out_contact_pts2, out_contact_distance


def apply_affine_to_points(pts_yx, M):
    '''
    对点集进行仿射变换
    :param pts: 要求输入格式为 [yx, yx, ...]
    :param M: 3x3 或 2x3 变换矩阵，可以用 affine_matrix_tool 生成自定义仿射矩阵
    :return:
    '''
    # 现在对矩阵乘法熟悉了，可以用矩阵乱搞了。
    pts_yx = np.asarray(pts_yx)
    M = np.asarray(M)
    assert pts_yx.ndim == 2 and pts_yx.shape[1] == 2, 'Error! Unknow pts_yx shape!'
    assert M.shape in [(2, 3), (3, 3)], 'Error! Unknow M shape!'

    pts_xy = pts_yx[:, ::-1]
    pts_xyz = np.concatenate([pts_xy, np.ones([pts_xy.shape[0], 1], dtype=pts_xy.dtype)], axis=1)

    if M.shape == (2, 3):
        M = np.concatenate([M, np.zeros([1, 3], dtype=M.dtype)], axis=0)
    out_pts_xyz = np.empty_like(pts_xyz)
    for i in range(len(pts_xyz)):
        out_pts_xyz[i] = M.transpose([0,1]) @ pts_xyz[i]
    out_pts_xy = out_pts_xyz[:, :2]
    out_pts_yx = out_pts_xy[:, ::-1]
    return out_pts_yx


def resize_points(points: np.ndarray, factor_hw=(1, 1), center_yx=(0, 0)):
    '''
    基于指定位置对点的坐标进行缩放
    :param points:      要求 points 为 np.ndarray 和 格式为 y1x1
    :param factor_hw:   缩放倍率，可以单个数字或元组
    :param center_yx:   默认以(0, 0)为原点进行缩放
    :return:
    '''
    assert isinstance(points, np.ndarray)
    assert points.ndim in [1, 2] and points.shape[-1] == 2

    if not isinstance(factor_hw, Sized):
        factor_hw = [float(factor_hw), float(factor_hw)]

    assert len(center_yx) == 2
    assert len(factor_hw) == 2

    center_yx = np.array(center_yx)
    factor_hw = np.array(factor_hw)

    ori_dtype = points.dtype

    points = np.asarray(points, np.float32)
    points -= center_yx
    points *= factor_hw
    points += center_yx
    points = np.asarray(points, ori_dtype)

    return points


if __name__ == '__main__':

    # 测试 get_shortest_link_pair
    pts1 = [[1, 1], [5, 5], [100, 100], [150, 100]]
    pts2 = [[3,3], [70, 90], [140, 180]]
    pair_pts1_id, pair_pts2_id, dists = get_shortest_link_pair(pts1, pts2, 100)
    for i, j, d in zip(pair_pts1_id, pair_pts2_id, dists):
        print(i, j, d)

    # 测试 apply_affine_to_points
    pts_yx = [[0, 0], ]

    R = make_rotate(90, (0, 10), img_hw=None)
    rot_pts_yx = apply_affine_to_points(pts_yx, R)
    assert np.allclose(rot_pts_yx[0], [-10, 10])
    print(rot_pts_yx)

    T = make_move((2, 5), None)
    t_pts_yx = apply_affine_to_points(pts_yx, T)
    assert np.allclose(t_pts_yx[0], [2, 5])
    print(t_pts_yx)

    S = make_scale((2, 4), (2, 10), None)
    s_pts_yx = apply_affine_to_points(pts_yx, S)
    assert np.allclose(s_pts_yx, [-2, -30])
    print(s_pts_yx)
