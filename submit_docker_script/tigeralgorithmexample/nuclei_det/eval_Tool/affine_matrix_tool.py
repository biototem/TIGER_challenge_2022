'''
仿射矩阵相关工具，注意这里的坐标系是窗口坐标系，左上角为原点，往右为X+，往下为Y+
并且注意，这里返回的矩阵是用于顶点顺序为 xy 的顶点的处理
new_xy1 = M @ old_xy1
'''

import numpy as np
import cv2
from typing import Union, Tuple, Iterable


# 旋转，平移，缩放，切变


# def _rot_mat(deg):
#     '''
#     绕Z轴旋转，来自 glm::eulerAngleZ
#     :param angle:
#     :return:
#     '''
#     rad = deg / 180. * np.pi
#     cosZ = np.cos(rad)
#     sinZ = np.sin(rad)
#
#     M = np.array([
#         [cosZ, sinZ, 0],
#         [-sinZ, cosZ, 0],
#         [0, 0, 1],
#     ], dtype=np.float32)
#     return M


def get_rotation_matrix_2d(angle, center_yx, scale):
    '''
    等效于 cv2.getRotationMatrix2D
    :param angle: 旋转角度
    :param scale: 缩放
    :param center_yx: 旋转点
    :return:
    '''
    angle = np.deg2rad(angle)
    a = scale * np.cos(angle)
    b = scale * np.sin(angle)
    M = [[a, b, (1-a)*center_yx[1] - b*center_yx[0]],
         [-b, a, b*center_yx[1] + (1-a)*center_yx[0]]]
    M = np.asarray(M, np.float32)
    return M


def make_rotate(angle=180., center_yx=(0.5, 0.5), img_hw: Union[None, Tuple]=(100, 100), dtype=np.float32):
    '''
    旋转
    :param angle: 顺时针旋转角度
    :param center_yx: 如果 img_hw 不为 None，则为百分比坐标，否则为绝对坐标
    :param img_hw: 图像大小，单位为像素
    :return:
    '''
    if img_hw is not None:
        center_yx = (img_hw[0] * center_yx[0], img_hw[1] * center_yx[1])

    R = np.eye(3, dtype=dtype)
    # 这里加个符号使其顺时针转动
    R[:2] = get_rotation_matrix_2d(angle=-angle, center_yx=center_yx, scale=1.)

    # T = np.eye(3, dtype=dtype)
    # T[0, 2] = center_yx[1]
    # T[1, 2] = center_yx[0]
    #
    # invT = np.copy(T)
    # invT[0, 2] = -invT[0, 2]
    # invT[1, 2] = -invT[1, 2]
    #
    # # 这矩阵太奇怪了，我从glm里面再弄个算法
    # R = T @ _rot_mat(angle) @ invT
    return R


def make_move(move_yx=(0.2, 0.2), img_hw: Union[None, Tuple]=(100, 100), dtype=np.float32):
    '''
    平移
    :param move_yx: 平移距离，当 img_hw 不是None时，单位为图像大小百分比，如果 img_hw 是 None，则为像素单位
    :param img_hw: 图像大小，单位为像素
    :return:
    '''
    move_yx = list(move_yx)
    if img_hw is not None:
        move_yx[1] = move_yx[1] * img_hw[1]
        move_yx[0] = move_yx[0] * img_hw[0]

    T = np.eye(3, dtype=dtype)
    T[0, 2] = move_yx[1]
    T[1, 2] = move_yx[0]
    return T


def make_scale(scale_yx=(2., 2.), center_yx=(0.5, 0.5), img_hw: Union[None, Tuple]=(100, 100), dtype=np.float32):
    '''
    缩放
    :param scale_yx: 单位必须为相对图像大小的百分比
    :param center_yx: 变换中心位置，当 img_hw 不是None时，单位为图像大小百分比，如果 img_hw 是 None，则为像素单位
    :param img_hw: 图像大小，单位为像素
    :return:
    '''
    if img_hw is not None:
        center_yx = (img_hw[0] * center_yx[0], img_hw[1] * center_yx[1])

    S = np.eye(3, dtype=dtype)
    S[0, 0] = scale_yx[1]
    S[1, 1] = scale_yx[0]
    S[0, 2] = (1-scale_yx[1]) * center_yx[1]
    S[1, 2] = (1-scale_yx[0]) * center_yx[0]

    return S


def make_shear(shear_yx=(1., 1.), dtype=np.float32):
    '''
    切变
    :param shear_yx: 单位为角度；变量1，图像x边与窗口x边角度，变量2，图像y边与窗口y边角度
    :return:
    '''
    # Shear
    S = np.eye(3, dtype=dtype)
    S[0, 1] = np.tan(shear_yx[1] * np.pi / 180)  # x shear (deg)
    S[1, 0] = np.tan(shear_yx[0] * np.pi / 180)  # y shear (deg)
    return S


if __name__ == '__main__':
    pt_xy = np.array([0, 0, 1], dtype=np.float32)

    R = make_rotate(90, (0, 10), img_hw=None)
    rot_pt_xy = R @ pt_xy
    assert np.allclose(rot_pt_xy, [10, -10, 1])
    print(rot_pt_xy)

    T = make_move((2, 5), None)
    t_pt_xy = T @ pt_xy
    assert np.allclose(t_pt_xy, [5, 2, 1])
    print(t_pt_xy)

    S = make_scale((2, 4), (2, 10), None)
    s_pt_xy = S @ pt_xy
    assert np.allclose(s_pt_xy, [-30, -2, 1])
    print(s_pt_xy)

    # pt_yx = np.array([2, 2, 1], dtype=np.float32)
    # Sh = make_shear((90, 90))
    # sh_pt_yx = Sh @ pt_yx
    # # assert np.allclose(sh_pt_yx, [0, -10, 1])
    # print(sh_pt_yx)