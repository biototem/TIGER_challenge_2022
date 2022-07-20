import numpy as np
import cv2
from nuclei_det.eval_Tool import contour_tool
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