'''
图像处理辅助库，作为 scikit-image 和 opencv 辅助用，非替代
'''

from typing import Iterable
import imageio
import numpy as np
import os
import cv2
from skimage.transform import resize as sk_resize
from PIL import Image, ImageDraw, ImageFont
from typing import Union, Tuple, Iterable


def check_and_tr_umat(mat):
    '''
    如果输入了cv.UMAT，将会自动转换为 np.ndarray
    :param mat:
    :return:
    '''
    if isinstance(mat, cv2.UMat):
        mat = mat.get()
    return mat


def check_and_tr_color_to_tuple(color_value: Union[int, float, Iterable], im_shape: Tuple[int]):
    '''
    如果输入了颜色标量，将会根据原图维度自动转换为合适的颜色元组
    例子:输入 color_value=50           im_shape=(100, 200, 3)，输出 (50, 50, 50)
         输入 color_value=50           im_shape=(100, 200, )，输出 50
         输入 color_value=(50,)        im_shape=(100, 200, 3)，报错
         输入 color_value=(50, 50, 50) im_shape=(100, 200, 3)，输出 (50, 50, 50)
         输入 color_value=(50, 50, 50) im_shape=(100, 200, )，报错
    :param color_value:
    :param im_shape:
    :return:
    '''
    assert len(im_shape) in {2, 3}
    if isinstance(color_value, Iterable):
        if len(im_shape) == 2:
            assert len(color_value) == 1
        else:
            # 不进行额外处理，直接抛出错误
            assert im_shape[-1] == len(color_value)
    else:
        if len(im_shape) != 2:
            color_value = (color_value) * im_shape[-1]

    return color_value


def ensure_image_has_3dim(im):
    '''
    确保图像有三个维度，opencv的函数经常会把灰度图像的最后的一个维度去掉
    :param im:
    :return:
    '''
    if im.ndim != 3:
        im = im[:, :, None]
    return im


def ensure_image_has_same_ndim(im: np.ndarray, ori_im: np.ndarray):
    '''
    确保图像与原图有相同的维度，opencv的函数经常会把灰度图像的最后的一个维度去掉
    :param im:      待处理的图像
    :param ori_im:  原图
    :return:
    '''
    if ori_im.ndim > im.ndim:
        # 比原图少
        assert im.ndim == 2
        im = im[:, :, None]
    elif ori_im.ndim < im.ndim:
        # 比原图多
        assert im.ndim == 3 and im.shape[-1] == 1
        im = im[:, :, 0]
    assert im.ndim == ori_im.ndim
    return im


def resize_image(img, target_hw, interpolation=cv2.INTER_LINEAR, *, use_sk_func=False):
    '''
    cv2.resize pack function
    :param img:
    :param target_hw:
    :param interpolation: for example cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_NEAREST or 
    :param use_sk_func: will use skimage.transform.resize to replace cv2.resize
    :return:
    '''
    if use_sk_func:
        out = sk_resize(img, target_hw, interpolation, mode='constant', cval=0, clip=True, preserve_range=True, anti_aliasing=False)
        out = np.asarray(out, img.dtype)
    else:
        out = cv2.resize(img, dsize=tuple(target_hw[::-1]), interpolation=interpolation)
        out = check_and_tr_umat(out)
        out = ensure_image_has_same_ndim(out, img)
    return out


def show_image(img, title='show_img'):
    """
    show image
    :param img: numpy array
    :return: None
    """
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow(title, img)
    cv2.waitKey(0)


def pad_picture(img, width, height, interpolation=cv2.INTER_NEAREST, fill_value: Union[int, float, Tuple]=0.):
    """
    padded picture to specified shape
    :param img: input numpy array
    :param width: output image width
    :param height: output image height
    :param interpolation: control img interpolation
    :param fill_value: control background fill value
    :return: output numpy array
    """
    s_height, s_width = img.shape[:2]
    new_shape = [height, width]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    # ratio = s_width / s_height
    width_prop = width / s_width
    height_prop = height / s_height
    min_prop = min(width_prop, height_prop)
    img2 = cv2.resize(img, (int(s_width * min_prop), int(s_height * min_prop)), interpolation=interpolation)
    img2 = ensure_image_has_same_ndim(img2, img)
    img = img2
    img_start_x = width / 2 - s_width * min_prop / 2
    img_start_y = height / 2 - s_height * min_prop / 2
    fill_value = check_and_tr_color_to_tuple(fill_value, new_shape)
    new_img = np.empty(new_shape, dtype=img.dtype)
    new_img[:, :] = np.asarray(fill_value)
    new_img[int(img_start_y):int(img_start_y)+img.shape[0], int(img_start_x):int(img_start_x)+img.shape[1]] = img
    return new_img


def center_pad(im, target_hw, fill_value: Union[int, float, Tuple]=0):
    '''
    中间填充，注意仅当原图小于目标宽高的时候才会进行填充，并且不会将生成的图缩放到目标宽高
    :param im:
    :param target_hw:
    :param fill_value:
    :return:
    '''
    pad_h = target_hw[0] - im.shape[0]
    pad_w = target_hw[1] - im.shape[1]
    pad_t = pad_h // 2
    pad_b = pad_h - pad_t
    pad_l = pad_w // 2
    pad_r = pad_w - pad_l
    if pad_h < 0:
        pad_t = pad_b = 0
    if pad_w < 0:
        pad_l = pad_r = 0
    fill_value = check_and_tr_color_to_tuple(fill_value, im.shape)
    im = cv2.copyMakeBorder(im, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=fill_value)
    return im


def crop_picture(img, width, height):
    """
    padded picture to specified shape
    :param img: input numpy array
    :param width: output image width
    :param height: output image height
    :return: output numpy array
    """
    s_height, s_width, s_depth = img.shape
    s_ratio = s_width / s_height
    ratio = width / height
    if s_ratio > ratio:
        need_crop_x = (s_width - s_height * ratio) / 2
        new_width = s_height * ratio
        img = img[0:s_height, int(need_crop_x):int(need_crop_x+new_width), :]
    elif s_ratio < ratio:
        need_crop_y = (s_height - s_width / ratio) / 2
        new_height = s_width / ratio
        img = img[int(need_crop_y):int(need_crop_y+new_height), 0:s_width, :]
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
    return img


# put_text需要指定字体，设定默认字体
_default_font_path = os.path.join(os.path.dirname(__file__), 'SourceHanSansCN-Regular.otf')

def put_text(img, text, pos, font_size=20, font_color=(0, 0, 255), bg_color=None, font_type=None):
    """
    draw font on image
    :param img: np.array
        input image
    :param text: str
        the text will draw on image
    :param pos: (x, y)
        where to draw text
    :param font_size: int
        like the name
    :param font_color: (r,g,b)
        0-255, like the name
    :param font_type: str
        which font would you want
    :return: np.array
        output image
    """
    if bg_color is not None and bg_color is not str:
        bg_color = tuple(bg_color)

    if font_type is None:
        font_type = _default_font_path

    pil_im = Image.fromarray(np.asarray(img, np.uint8))
    draw = ImageDraw.Draw(pil_im)
    font = ImageFont.truetype(font_type, int(font_size))
    w, h = draw.multiline_textsize(text, font)
    draw.rectangle((pos[0], pos[1], pos[0]+w, pos[1]+h), bg_color)
    draw.multiline_text(pos, text, tuple(font_color), font=font)
    return np.asarray(pil_im, np.uint8)


_start_color = np.array([64, 128, 192])
_color_step = np.array([173, 79, 133])

def get_random_color():
    """
    Get random color
    :return: np.array([r,g,b])
    """
    global _start_color, _color_step
    # rgb = np.random.uniform(0, 25, [3])
    # rgb = np.asarray(np.floor(rgb) / 24 * 255, np.uint8)
    _start_color = (_start_color + _color_step) % np.array([256, 256, 256])
    rgb = np.asarray(_start_color, np.uint8).tolist()
    return rgb


def draw_keypoints_and_labels_to_image_coco(image, keypoints, skeleton, keypoints_name_list):
    """
    draw keypoints with coco data type
    :param image:
    :param keypoints:
    :param skeleton:
    :param keypoints_name_list:
    :return:
    """
    image = image.copy()
    imh, imw = image.shape[0:2]
    thick = int((imh + imw) // 500)
    for k, kpt in enumerate(keypoints):
        kpt = np.asarray(kpt, np.int32)
        x, y, v = kpt
        if kpt[2] == 0:
            continue
        cv2.circle(image, (x, y), max(int(1.5e-3 * imh), 1), (0, 0, 255), -1)
        text = keypoints_name_list[k]
        cv2.putText(image, text, (x, y), 0, 1.5e-3 * imh, [0, 0, 255], int(thick / 3) + 1)
    for s in skeleton:
        x, y, v = keypoints[s[0]]
        x2, y2, v2 = keypoints[s[1]]
        if v > 0 and v2 > 0:
            cv2.line(image, (x, y), (x2, y2), [255, 0, 0], thick)
    return image


def draw_boxes_and_labels_to_image(image, classes, coords, scores=None, classes_list=None, classes_colors=None, font_color=(0, 0, 255)):
    """
    Draw bboxes and class labels on image. Return or save the image with bboxes
    Parameters
    -----------
    image : numpy.array
        The RGB image [height, width, channel].
    classes : list of int
        A list of class ID (int).
    coords : list of int
        A list of list for coordinates.
            - Should be [x, y, x2, y2]
    scores : list of float
        A list of score (float). (Optional)
    classes_list : list of str
        For converting ID to string on image.
    classes_colors : list of color
        A list of color [ [r,g,b], ...].
    font_color : front color
        Front color
    Returns
    -------
    numpy.array
        The output image.
    """
    image = image.copy()
    imh, imw = image.shape[0:2]
    thick = int((imh + imw) // 500)     # 粗细
    for i, _v in enumerate(coords):
        x, y, x2, y2 = np.asarray(coords[i], np.int32)
        bbox_color = [0, 255, 0] if classes_colors is None else classes_colors[classes[i]]
        cv2.rectangle(image, (x, y), (x2, y2), bbox_color, thick)
        if classes is not None:
            class_text = classes_list[classes[i]] if classes_list is not None else str(classes[i])
            score_text = " %.2f" % (scores[i]) if scores is not None else ''
            text = class_text + score_text

            font_scale = 1.0e-3 * imh
            # text_size, _ = cv2.getTextSize(text, 0, font_scale, int(thick / 2) + 1)
            # cv2.rectangle(image, (x, y), (x+text_size[0], y-text_size[1]), bbox_color, -1)
            # cv2.putText(image, text, (x, y), 0, font_scale, font_color, int(thick / 3) + 1)
            image = put_text(image, text, (x, y), font_scale*32, font_color, bbox_color)
    return image


# def extract_2d_patches_with_half_step(im, patch_hw, need_center_pad=False):
#     '''
#     :param im:
#     :param patch_hw:
#     :return:
#     '''
#     assert im.shape[0] % patch_hw[0] == 0 and im.shape[1] % patch_hw[1] == 0
#     half_step_h = patch_hw[0] // 2
#     half_step_w = patch_hw[1] // 2
#     im_patches = []
#     for half_h in range(im.shape[0] // half_step_h - 1):
#         for half_w in range(im.shape[1] // half_step_w - 1):
#             im_patches.append(im[half_h*half_step_h: half_h*half_step_h+patch_hw[0],
#                                  half_w*half_step_w: half_w*half_step_w+patch_hw[1]])
#     return np.asarray(im_patches)


def draw_multi_img_in_big_img(imgs, out_dim, im_hw, n_hw, pad_color=0):
    '''
    Draw multiple images into one big image.
    :param imgs:        Multiple images. The number of channels is not required to match.
    :param out_dim:     The number of channels for the big image.
    :param im_hw:       The height and width of the big image.
    :param n_hw:        How many images are placed in each row and column.
    :param pad_color:   The color of the fill.
    :return: A big image.
    '''
    dtype = np.uint8
    if len(imgs) > 0:
        dtype = imgs[0].dtype
    big_im = np.empty([*im_hw, out_dim], np.uint8)
    big_im[..., :] = pad_color
    each_im_hw = (im_hw[0] // n_hw[0], im_hw[1] // n_hw[1])
    n_max_show = np.prod(n_hw)
    for i in range(min(len(imgs), n_max_show)):
        im = imgs[i]

        # Make the number of channels equal.
        im = ensure_image_has_3dim(im)
        if im.shape[2] > out_dim:
            im = im[..., :out_dim]
        elif im.shape[2] < out_dim:
            t = np.zeros([im.shape[0], im.shape[1], out_dim], dtype=im.dtype)
            t[..., : im.shape[2]] = im
            im = t

        # Scale each image to the appropriate size.
        im = pad_picture(im, each_im_hw[1], each_im_hw[0], cv2.INTER_AREA, pad_color)
        im = ensure_image_has_3dim(im)

        ph = i // n_hw[1]
        pw = i % n_hw[1]
        big_im[ph * each_im_hw[0]: (ph + 1) * each_im_hw[0], pw * each_im_hw[1]: (pw + 1) * each_im_hw[1]] = im
    return big_im


def copy_make_border(src, top, bottom, left, right, value=None):
    '''
    My implement cv2.copyMakeBorder.
    自己实现 cv2.copyMakeBorder 以避免错误 TypeError: Scalar value for argument 'value' is longer than 4.
    :param src:     input image
    :param top:
    :param bottom:
    :param left:
    :param right:
    :param value:   pad value
    :return:
    '''
    assert top >= 0 and bottom >= 0 and left >= 0 and right >= 0, 'top, bottom, left or right must be equal or bigger than 0.'
    new_im = np.zeros([src.shape[0] + top + bottom, src.shape[1] + left + right, src.shape[2]], dtype=src.dtype)
    nh, nw = new_im.shape[:2]
    if value is not None:
        new_im[..., :] = value
    new_im[top: nh-bottom, left: nw-right] = src
    return new_im


def bin_infill(im: np.ndarray):
    '''
    二进制填充函数
    :param im:
    :return:
    '''
    assert im.ndim == 3
    new_im = np.zeros_like(im, dtype=np.uint8)
    for i in range(im.shape[2]):
        cons, _ = cv2.findContours((im[:, :, i] != 0).astype(np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        new_im[:, :, i] = cv2.drawContours(cv2.UMat(new_im[:, :, i]), cons, -1, 1, -1).get()
    return new_im


def pad_img_to_target_ratio(im, ratio=1.):
    '''
    居中填充图像到指定比例，并且返回填充参数
    :param im:
    :param ratio: 比例格式为h/w
    :return:
    '''
    assert im.ndim == 3

    h, w = im.shape[0:2]
    if h / w > ratio:
        w = int(h / ratio)
    elif h / w < ratio:
        h = int(w * ratio)

    pad_h = h - im.shape[0]
    pad_h_1 = pad_h // 2
    pad_h_2 = pad_h - pad_h_1

    pad_w = w - im.shape[1]
    pad_w_1 = pad_w // 2
    pad_w_2 = pad_w - pad_w_1

    im = np.pad(im, [[pad_h_1, pad_h_2], [pad_w_1, pad_w_2], [0, 0]])
    param = [[pad_h_1, pad_h_2], [pad_w_1, pad_w_2]]
    return im, param


def resize_img_to_target_size(im: np.ndarray, size_hw, inter=cv2.INTER_AREA):
    '''
    缩放图像到指定大小，并且返回缩放参数
    :param im:
    :param size_hw:
    :return:
    '''
    assert im.ndim == 3
    assert len(size_hw) == 2
    im_hw = np.float32(im.shape[:2])
    size_hw = np.float32(size_hw)
    param = size_hw / im_hw
    oim = cv2.resize(im, tuple(np.int32(size_hw[::-1])), interpolation=inter)
    oim = ensure_image_has_3dim(oim)
    return oim, param


def show_hm_on_image(img: np.ndarray,
                     hm: np.ndarray,
                     max_blend_value: float = 0.3,
                     colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    '''
    在图像上绘制热图，便于观察
    :param img: 输入图像，要求为RGB图形，shape 为 (H, W, 3)，dtype 为 np.uint8
    :param hm:  输入热图，要求为灰度图，shape 为 (H, W) or (H, W, 1)，dtype 为 np.float16 or np.float32 or np.float64
    :param max_blend_value: 最大混合值，默认为 0.3
    :param colormap: 颜色映射方法，默认为 cv2.COLORMAP_JET
    :return:
    '''
    assert isinstance(img, np.ndarray)
    assert isinstance(hm, np.ndarray)
    assert img.dtype == np.uint8
    assert img.ndim == 3 and img.shape[-1] == 3
    assert hm.dtype in (np.float16, np.float32, np.float64)
    assert hm.ndim in [2, 3] and (hm.ndim != 3 or hm.shape[-1] == 1)

    hm = ensure_image_has_3dim(hm)

    color_hm = cv2.applyColorMap(np.uint8(255 * hm), colormap)
    color_hm = cv2.cvtColor(color_hm, cv2.COLOR_BGR2RGB)

    img = np.float32(img)
    hm = np.float32(hm)
    color_hm = np.float32(color_hm)

    alpha = np.minimum(hm, max_blend_value)

    oim = img * (1-alpha) + color_hm * alpha
    oim = np.uint8(np.round(oim).clip(0, 255))
    return oim


def test():
    mod_dir = os.path.dirname(__file__)

    im = imageio.imread(mod_dir + '/laska.png')

    # test pad_picture
    new_img = pad_picture(im, 1024, 700)
    show_image(np.asarray(new_img, np.uint8))

    new_img = pad_picture(im, 700, 1024, fill_value=[255, 255, 255, 0])
    show_image(np.asarray(new_img, np.uint8))

    # test center_pad
    new_img = center_pad(im, [700, 700])
    show_image(np.asarray(new_img, np.uint8))

    new_img = center_pad(im, [700, 700], fill_value=[128, 128, 128, 0])
    show_image(np.asarray(new_img, np.uint8))

    # test crop_picture
    new_img = crop_picture(im, 1024, 512)
    show_image(np.asarray(new_img, np.uint8))

    new_img = crop_picture(new_img, 512, 1024)
    show_image(np.asarray(new_img, np.uint8))

    # test put_text
    new_img = pad_picture(im, 512, 512)
    new_img = put_text(new_img, '你好world', (256, 256))
    show_image(np.asarray(new_img, np.uint8))

    # test get_random_color
    m = np.zeros([16 * 16, 3], np.uint8)
    for i in range(16 * 16):
        m[i] = get_random_color()
    m = np.reshape(m, (16, 16, 3))
    m = cv2.resize(m, (256, 256), interpolation=cv2.INTER_NEAREST)
    # m = np.asarray(resize(m, (256, 256, 3), 0, 'constant', preserve_range=True, anti_aliasing=False), np.uint8)
    show_image(m)

    # test draw_multi_img_in_big_img
    new_img = draw_multi_img_in_big_img([im]*16, 3, [512, 768], [4, 4], (256, 128, 64))
    show_image(new_img)

    # test copy_make_border
    new_img = copy_make_border(im, 10, 30, 50, 70, [20, 40, 60, 0])
    show_image(new_img)

    # test bin_infill
    new_img_1 = np.zeros([512, 512, 3], np.uint8)
    new_img_1[:, :, 0] = cv2.rectangle(cv2.UMat(new_img_1[:, :, 0]), (100, 100), (200, 200), 255, 2).get()
    new_img_1[:, :, 1] = cv2.rectangle(cv2.UMat(new_img_1[:, :, 1]), (150, 150), (250, 250), 128, 2).get()
    new_img_2 = bin_infill(new_img_1)
    show_image(new_img_2 * 255)


if __name__ == '__main__':
    test()
