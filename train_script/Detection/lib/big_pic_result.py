# import os
# if os.name == 'nt':
#     openslide_bin = os.path.abspath(os.path.split(__file__)[0]+'/../bin_openslide_x64_20171122')
#     os.putenv('PATH', os.getenv('PATH') + ';' + openslide_bin)

# import openslide
import cv2
import imageio
import numpy as np
from typing import Iterable, Callable

from lib.coords_over_scan_gen import n_step_scan_coords_gen
from lib.image_over_scan_wrapper import ImageOverScanWrapper
from lib.universal_batch_generator import universal_batch_generator
import lib.contour_tool as contour_tool
import lib.im_tool as im_tool
import eval_utils


class BigPicPatch:
    '''
    现在对这个进行简化，仅负责图像的管道处理和储存
    大图块就是这里
    现在加入缓存机制，重复读取时将会使用缓存
    '''
    def __init__(self, n_class, multi_scale_img, top_left_point, window_hw, level_0_patch_hw, ignore_contour_near_border_ratio=0.001, ignore_patch_near_border_ratio=0.1, tissue_contours=None, *, multi_scale_result=None, multi_scale_mask=None, custom_patch_merge_pipe=None, custom_gen_contours_pipe=None, patch_border_pad_value=0):
        '''
        :param n_class:                             类别数量
        :param multi_scale_img:                     多个尺度的图像，注意，以第一个图像为基准，其他的图都会以它进行缩放
        :param top_left_point:                      此大图块的左上角起始坐标
        :param window_hw:                           在大图块中的滑窗大小
        :param level_0_patch_hw:                    图块的原始大小，当多个尺度图像不包含0级图时，可以缩放到0级图尺度
        :param ignore_contour_near_border_ratio:    忽略靠近边缘的轮廓，如果轮廓与边缘的最小距离小于该比例，则抛弃该轮廓
        :param ignore_patch_near_border_ratio:      忽略每一个小滑窗patch的边缘比例
        :param tissue_contours:                     level_0尺度下的mask轮廓，可以加速处理，不在轮廓内的滑窗将会被跳过
        :param multi_scale_result:                  用来恢复缓存的
        :param custom_patch_merge_pipe:             自定义滑窗合并方法，如不设定，self._default_patch_merge是默认的处理方法
        :param custom_gen_contours_pipe:            自定义轮廓生成方法，如不设定，self._default_get_contours是默认的处理方法
        '''
        self.multi_scale_img = [ImageOverScanWrapper(im) for im in multi_scale_img]
        self.top_left_point = np.asarray(top_left_point, np.int)  # (0, 0)
        self.n_class = n_class  # 7
        self.window_hw = window_hw   # (64, 64)
        self.level_0_patch_hw = np.asarray(level_0_patch_hw, np.int)   # (1, 1)
        self.ignore_contour_near_border_ratio = ignore_contour_near_border_ratio  # 0.001
        self.ignore_patch_ratio = ignore_patch_near_border_ratio    # 0.5
        self.tissue_contours = tissue_contours   # None
        self.patch_border_pad_value = patch_border_pad_value    # 255

        # 缓存标记
        self._contours_cache_data = None
        self._contours_need_update = True
        self._need_heatmap_fresh = True
        self._cache_heatmap = None

        # 多尺度结果
        if multi_scale_result is None:
            multi_scale_result = [np.zeros([*im.shape[:2], n_class], np.float32) for im in multi_scale_img]
        else:
            for i, _ in enumerate(multi_scale_img):
                H, W, C = multi_scale_result[i].shape
                assert H == multi_scale_img[i].shape[0] and W == multi_scale_img[i].shape[1] and C == n_class and multi_scale_result[i].dtype == np.float32
        self.multi_scale_result = [ImageOverScanWrapper(im) for im in multi_scale_result]

        # 多尺度掩码，用来记录每一副图的每个部分是否被处理过，目前只用于自定义patch融合管道和自定义轮廓生成管道，默认管道不使用
        if multi_scale_mask is None:
            multi_scale_mask = [np.zeros([*im.shape[:2], n_class], np.uint8) for im in multi_scale_img]
        else:
            for i, _ in enumerate(multi_scale_mask):
                H, W, C = multi_scale_mask[i].shape
                assert H == multi_scale_img[i].shape[0] and W == multi_scale_img[i].shape[1] and C == n_class and multi_scale_mask[i].dtype == np.uint8
        self.multi_scale_mask = [ImageOverScanWrapper(im) for im in multi_scale_mask]

        # 检查和设定自定义方法
        assert custom_patch_merge_pipe is None or isinstance(custom_patch_merge_pipe, Callable)
        assert custom_gen_contours_pipe is None or isinstance(custom_gen_contours_pipe, Callable)
        self.custom_patch_merge_pipe = custom_patch_merge_pipe
        self.custom_gen_contours_pipe = custom_gen_contours_pipe

    def _get_pic_patch(self, level, yx_start, yx_end, pad_value):
        out_im = self.multi_scale_img[level].get(yx_start, yx_end, pad_value)
        return out_im

    def _set_pic_patch(self, level, yx_start, yx_end, patch_result_new):
        # 更新缓存标记     patch_result_new： (64, 64, 7)存储的就是三个分支的模型输出结果
        self._contours_need_update = True
        self._need_heatmap_fresh = True
        patch_result_cur = self.multi_scale_result[level].get(yx_start, yx_end)  # 7 个通道全为0
        patch_mask_cur = self.multi_scale_mask[level].get(yx_start, yx_end)  # 7 个通道全为0
        assert patch_result_new.ndim == patch_result_cur.ndim and patch_result_new.shape[-1] == patch_result_cur.shape[-1]
        assert patch_result_new.shape[0] == yx_end[0] - yx_start[0] and patch_result_new.shape[1] == yx_end[1] - yx_start[1]

        # 处理忽略边缘问题
        ignore_edge_hw_pixel = (np.array(patch_result_new.shape[:2]) * self.ignore_patch_ratio / 2).astype(np.int) # [16,16]
        ys = ignore_edge_hw_pixel[0] # 16
        ye = None if ignore_edge_hw_pixel[0] == 0 else -ignore_edge_hw_pixel[0] # -16
        xs = ignore_edge_hw_pixel[1] # 16
        xe = None if ignore_edge_hw_pixel[1] == 0 else -ignore_edge_hw_pixel[1] # -16

        if self.custom_patch_merge_pipe is None:
            new_result, new_mask = self._default_patch_merge(self, patch_result_cur.copy(), patch_result_new.copy(), patch_mask_cur.copy())
        else:
            new_result, new_mask = self.custom_patch_merge_pipe(self, patch_result_cur.copy(), patch_result_new.copy(), patch_mask_cur.copy())

        patch_result_cur[ys: ye, xs: xe] = new_result[ys: ye, xs: xe]
        patch_mask_cur[ys: ye, xs: xe] = new_mask[ys: ye, xs: xe]

        self.multi_scale_result[level].set(yx_start, yx_end, patch_result_cur)
        self.multi_scale_mask[level].set(yx_start, yx_end, patch_mask_cur)

    def get_im_patch_gen(self):
        for level, im in enumerate(self.multi_scale_img):
            for yx_start, yx_end in n_step_scan_coords_gen(im.shape[:2], self.window_hw, 2):
                # 这里代码应该没问题了
                # 预先判断，可以跳过不必要的运算
                if self.tissue_contours is not None:
                    bbox = np.asarray([*yx_start, yx_start[0] + self.window_hw[0], yx_start[1] + self.window_hw[1]])
                    bbox_cont = contour_tool.make_contour_from_bbox(bbox)
                    factor_hw = self.level_0_patch_hw / im.shape[:2]
                    bbox_cont = contour_tool.resize_contours([bbox_cont], factor_hw)[0]
                    bbox_cont = contour_tool.offset_contours([bbox_cont], (0, 0), self.top_left_point)[0]
                    ious = contour_tool.calc_iou_with_contours_1toN(bbox_cont, self.tissue_contours)
                    if np.max(ious) == 0:
                        continue

                out_im = self._get_pic_patch(level, yx_start, yx_end, self.patch_border_pad_value)
                # # 第二个加速功能，判断图像白色像素（3通道均大于215的像素）占比是否大于95%，若大于则跳过
                # bm1 = np.all(out_im > np.reshape([215, 215, 215], [1, 1, 3]), -1)
                # if bm1.sum(dtype=np.int) * 0.95 > bm1.shape[0] * bm1.shape[1]:
                #     continue

                info = (level, yx_start, yx_end)
                yield info, out_im

    def update_multi_scale_result(self, info, result_im):
        level, yx_start, yx_end = info
        assert result_im.shape[0] == self.window_hw[0] and result_im.shape[1] == self.window_hw[1]
        self._set_pic_patch(level, yx_start, yx_end, result_im)

    def batch_get_im_patch_gen(self, batch_size):
        # 上面 get_pic_gen 的批量版，用来加速
        g = self.get_im_patch_gen()
        return universal_batch_generator(g, batch_size)

    def batch_update_result(self, batch_info, batch_result):
        assert len(batch_info) == len(batch_result)
        for p in zip(batch_info, batch_result):
            self.update_multi_scale_result(*p)

    def get_final_prob_heatmap(self, scale_to_level_0=False):
        '''
        获取平均融合后的热图，注意这里仅用于观察
        :param scale_to_level_0:
        :return:
        '''
        # 使用缓存
        if not self._need_heatmap_fresh:
            biggest_result = self._cache_heatmap
            if scale_to_level_0 and np.any(biggest_result.shape[:2] != self.level_0_patch_hw):
                biggest_result = im_tool.resize_image(biggest_result, self.level_0_patch_hw, interpolation=cv2.INTER_LINEAR)
            return biggest_result

        # 先对不同尺度的概率热图单独进行后处理，然后再合并，可以避免不同尺度后处理方式不同的问题
        biggest_result = self.multi_scale_result[0].data.copy()
        biggest_result = eval_utils.prob_heatmap_post_pro(biggest_result)
        for scale_im in self.multi_scale_result[1:]:
            scale_im = eval_utils.prob_heatmap_post_pro(scale_im.data)
            biggest_result += cv2.resize(scale_im, biggest_result.shape[:2][::-1], interpolation=cv2.INTER_LINEAR)

        biggest_result /= len(self.multi_scale_result)

        # 生成缓存
        self._need_heatmap_fresh = False
        self._cache_heatmap = biggest_result

        # 这里使用自我递归技巧避免再写一次缩放，最多只递归一次
        if scale_to_level_0:
            return self.get_final_prob_heatmap(scale_to_level_0=True)

        return biggest_result

    def get_draw_contours_im(self, class_id_to_color_tuple: dict):
        '''
        获取一张绘制好轮廓的图像
        :param class_id_to_color_tuple:
        :return:
        '''
        contours, clss = self.get_contours(dont_use_start_yx=True, dont_resize_contours=False)
        # 如果报错，则输出更多提示信息
        assert set(clss).issubset(class_id_to_color_tuple.keys()),\
            'Error! Found class ids not in class_id_to_color_tuple.keys() {} / {}'.format(
            set(clss),
            class_id_to_color_tuple.keys())
        factor = np.asarray(self.multi_scale_img[0].shape[:2], np.float32) / self.level_0_patch_hw
        contours = contour_tool.resize_contours(contours, factor)
        tmp = dict()
        for con, cls in zip(contours, clss):
            if class_id_to_color_tuple[cls] not in tmp:
                tmp[class_id_to_color_tuple[cls]] = []
            tmp[class_id_to_color_tuple[cls]].append(con)

        im = self.multi_scale_img[0].data
        for color in tmp:
            im = contour_tool.draw_contours(im, tmp[color], color, thickness=3)
        return im

    @staticmethod
    def _default_patch_merge(self, patch_result, patch_result_new, patch_mask):
        '''
        默认的滑窗图合并函数，合并时取最大值
        :param self:                引用大图块自身，用于实现某些特殊用途，一般不使用
        :param patch_result:        当前滑窗区域的结果
        :param patch_result_new:    新的滑窗区域的结果
        :param patch_mask:          当前掩码，用于特殊用途，这里不使用
        :return: 返回合并后结果和更新的掩码
        '''
        new_result = np.maximum(patch_result, patch_result_new)
        new_mask = patch_mask
        return new_result, new_mask

    @staticmethod
    def _default_gen_contours(self, multi_scale_result, multi_scale_mask):
        '''
        默认的轮廓生成函数，当前只是简单合并热图，然后以0.5做阈值，然后生成轮廓
        :param self:                引用大图块自身，用于实现某些特殊用途，一般不使用
        :param multi_scale_result:  多尺度结果图
        :param multi_scale_mask:    多尺度mask图
        :return: 返回轮廓和每个轮廓的类别
        '''
        prob = 0.5
        top_result = multi_scale_result[0]
        top_result_hw = top_result.shape[:2]

        for ms in multi_scale_result[1:]:
            ms = im_tool.resize_image(ms, top_result.shape[:2], interpolation=cv2.INTER_LINEAR)
            top_result = np.maximum(top_result, ms)
        contours = []
        clss = []
        top_result = (top_result > prob).astype(np.uint8)
        for c in range(top_result.shape[2]):
            cs = contour_tool.find_contours(top_result[:, :, c], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours.extend(cs)
            clss.extend([c] * len(cs))
        return top_result_hw, contours, clss

    def get_contours(self, dont_use_start_yx=False, dont_resize_contours=False):
        '''
        获取轮廓
        :param dont_use_start_yx:       若为True，则返回的轮廓坐标不会偏移
        :param dont_resize_contours:    若为True，则返回的轮廓不会缩放到level_0尺度
        :return:
        '''
        if self._contours_need_update:
            multi_scale_result = [d.data for d in self.multi_scale_result]
            multi_scale_mask = [d.data for d in self.multi_scale_mask]
            if self.custom_gen_contours_pipe is None:
                top_result_hw, contours, clss = self._default_gen_contours(self, multi_scale_result, multi_scale_mask)
            else:
                top_result_hw, contours, clss = self.custom_gen_contours_pipe(self, multi_scale_result, multi_scale_mask)

            need_remove_ids = eval_utils.find_contours_if_near_border(contours, top_result_hw, self.ignore_contour_near_border_ratio)
            for i in sorted(need_remove_ids)[::-1]:
                del contours[i]
                del clss[i]

            self._contours_cache_data = [top_result_hw, contours, clss]
            self._contours_need_update = False
        else:
            top_result_hw, contours, clss = self._contours_cache_data

        top_result_hw = np.asarray(top_result_hw)

        # 缩放轮廓尺寸为0级尺寸
        if not dont_resize_contours:
            if np.any(top_result_hw != self.level_0_patch_hw):
                contours = contour_tool.resize_contours(contours, self.level_0_patch_hw / top_result_hw)

        # 偏移轮廓
        if not dont_use_start_yx:
            contours = contour_tool.offset_contours(contours, (0, 0), self.top_left_point)

        return contours, clss


if __name__ == '__main__':
    print("aaa")