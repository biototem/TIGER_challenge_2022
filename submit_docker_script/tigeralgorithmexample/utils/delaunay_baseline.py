import math

import cv2
from scipy.spatial import Delaunay

from utils.opencv_utils import OpenCV
import numpy as np


'''
基于三角剖分基线法获取肿瘤区域矩阵
cancer_only_matrix：肿瘤矩阵
mask：组织区域为1，非组织区域为0
triangle_circle: 三角剖分算法中三角形外切圆半径
filter_rate：过滤肿瘤区域占矩阵区域的比率小于等于filter_rate的肿瘤区域
'''
def  get_cancer_region_baseline(cancer_only_matrix, mask, triangle_circle=75, filter_rate=0.1):

    contours = OpenCV(mask).find_contours(is_binary=True, mode='RETR_EXTERNAL')

    # 初始化整体肿瘤区域
    result_matrix = np.full((cancer_only_matrix.shape[0], cancer_only_matrix.shape[1]), 0, dtype=np.uint8)

    contours_len = len(contours)
    #
    # # 获取当前组织区域的肿瘤区域矩阵
    #
    filter_matrix_list = map(fill_contours,[cancer_only_matrix.shape]*contours_len,[contours])
    #
    part_cancer_only_matrix_list = map(lambda x:x* cancer_only_matrix,filter_matrix_list)
    # 利用三角剖分算法获取癌症区域矩阵
    cancer_matrix_list = list(map(get_triangle_matrix_baseline, part_cancer_only_matrix_list, [triangle_circle] , [10]*contours_len))
    # 过滤肿瘤区域面积占矩阵面积比率小于指定过滤值的肿瘤区域，cv2.RETR_EXTERNAL是为了填补肿瘤区域内的空洞
    cancer_matrix_list = list(map(filter_small_contour_region, cancer_matrix_list, [filter_rate]*contours_len))
    for i in cancer_matrix_list:
        result_matrix += i

    return result_matrix


'''
根据三角形的外切圆大小保留符合要求的坐标点
matrix_0: 二值化矩阵
triangle_circle: 外切圆大小
discrete_distance：离散间距，将矩阵进行离散化处理,这样可以大大加快三角剖分算法和三角填充的处理速度，当前默认为10
'''
def get_triangle_matrix_baseline(matrix_0: object, triangle_circle: object, discrete_distance: object) -> object:
    def add_edge(edges, edge_points, coords, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            return
        edges.add((i, j))
        edge_points.append([coords[i], coords[j]])

    matrix = np.copy(matrix_0)
    triangle_matrix = np.full((matrix.shape[0], matrix.shape[1]), 0, dtype=np.uint8)

    # 将矩阵进行离散化处理,这样可以大大加快三角剖分算法和三角填充的处理速度
    # 加速的主要原理是减少输入三角剖分算法的点坐标数量

    # 1.如果用于其他项目或者改变了矩阵下采样率，离散间距则需要进行调节，否则会造成三角剖分算法结果面积偏小。
    # 2.一般来说，若离散间距偏大，三角剖分面积便会偏小。这主要是因为过滤的点数偏多，影响三角剖分结果的轮廓边缘部分。
    # 3.当矩阵下采样率受到变化，亦或者是其他项目使用此函数时，应当尝试对比离散化处理和没离散化处理的三角剖分结果。
    # 4.以下使用10作为离散间距仅适用于当前结直肠癌项目九分类矩阵的下采样率。

    point_filter = np.full((matrix.shape[0], matrix.shape[1]), 0, dtype=np.uint8)
    point_filter[::discrete_distance, ::discrete_distance] = 1
    matrix = point_filter * matrix

    if np.max(matrix) != 0:

        # 获取所有需要三角剖分算法处理的点坐标
        npy = np.argwhere(matrix != 0)
        coords = [(i[0], i[1]) if type(i) or tuple else i for i in npy]
        if npy.size > 0:
            # 坐标位置调换，为了适应三角剖分算法的坐标位置
            # concave_hull, edge_points = alpha_shape(npy, 000.7)
            # concave = [concave_hull]
            npy[:, [0, 1]] = npy[:, [1, 0]]

            edges = set()
            edge_points = []
            eadge_npy2 = []
            # 三角剖分算法处理concave_hull
            tri = Delaunay(npy)
            for ia, ib, ic in tri.vertices:
                pa = coords[ia]
                pb = coords[ib]
                pc = coords[ic]
                # Lengths of sides of triangle
                a = math.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
                b = math.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
                c = math.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
                # Semiperimeter of triangle
                s = (a + b + c) / 2.0
                # Area of triangle by Heron's formula
                # 三角形面积
                area = math.sqrt(s * (s - a) * (s - b) * (s - c))
                # 三角形外切圆半径
                circum_r = a * b * c / (4.0 * area)
                # Here's the radius filter.
                # print circum_r
                if circum_r < triangle_circle:
                    add_edge(edges, edge_points, coords, ia, ib)
                    add_edge(edges, edge_points, coords, ib, ic)
                    add_edge(edges, edge_points, coords, ic, ia)
                    eadge_npy2.append(np.array([npy[ia], npy[ib], npy[ic]]))
            c = np.array(eadge_npy2)

            # 绘制三角形
            cv2.fillPoly(triangle_matrix, c, 1)

    return triangle_matrix


'''
填充轮廓到二维矩阵中，轮廓区域为1，非轮廓区域为0
shape：二维矩阵的shape
cnt_list：轮廓列表
'''
def fill_contours(shape, cnt_list):
    result_matrix = np.full((shape[0], shape[1]), 0, dtype=np.uint8)
    cv2.drawContours(result_matrix, cnt_list, -1, 1, cv2.FILLED)
    return result_matrix


'''
过滤轮廓面积占矩阵面积比率小于指定过滤值的轮廓区域
matrix_0：二值化矩阵
filter_rate：过滤值
Contour_Retrieval_Mode：默认cv2.RETR_TREE
'''
def filter_small_contour_region(matrix_0, filter_rate):
    matrix = np.copy(matrix_0)

    # 获取轮廓
    # contours=get_contours(matrix,Contour_Retrieval_Mode)

    # 初始化整体区域
    # result_matrix = np.full((matrix.shape[0], matrix.shape[1]), 0, dtype=matrix.dtype)、

    # # # 遍历所有轮廓区域
    # for cnt in contours:
    #
    #     # 获取当前区域面积
    #     area = cv2.contourArea(cnt)
    #
    #     # 获取当前区域占矩阵区域的面积比率
    #     rate_area = area / (matrix.shape[0] * matrix.shape[1])
    #
    #     # 过滤当前区域占矩阵区域的面积比率小于等于filter_rate的区域
    #     if rate_area > filter_rate:
    #         # 将当前区域过滤matrix，并合并到整体区域中
    #         result_matrix += fill_contours(matrix.shape, [cnt])

    # 获取轮廓
    contours = OpenCV(matrix).find_contours(is_binary=True, mode=0)
    # 初始化整体区域
    result_matrix = np.full((matrix.shape[0], matrix.shape[1]), 0, dtype=matrix.dtype)
    # 获取当前区域面积
    area_cnt_list = list(map(cv2.contourArea, contours))
    # # 获取当前区域占矩阵区域的面积比率
    # rate_area = [area / matrix.size for area in area_cnt_list]
    # # 获取当前区域占最大的面积比率
    rate_area = [area / max(area_cnt_list) for area in area_cnt_list]
    # 过滤当前区域占矩阵区域的面积比率小于等于filter_rate的区域
    rate_filter_area = [fill_contours(matrix.shape, [contours[i]])
                        for i in range(len(contours)) if rate_area[i] > filter_rate]
    # 将当前区域过滤matrix，并合并到整体区域中
    for i in rate_filter_area:
        result_matrix += i
    return result_matrix
