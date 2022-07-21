'''
N分之一步长坐标生成器，建议组合 ImageOverScanWrapper 来使用
'''


def n_step_scan_coords_gen(im_hw=(512, 512), window_hw=(128, 128), n_step: int=2):
    '''
    N分之一步长坐标生成器，可以生成滑动框左上角和右下角坐标，便于快速遍历图像
    当前只支持生成整数坐标
    :param im_hw:       图像大小
    :param window_hw:   窗口高宽
    :param n_step:      N分之一步长，默认为半步长
    :return: yx_start, yx_end   扫描框左上角和右下角坐标
    '''
    assert n_step >= 1
    window_step_hw = [window_hw[0] // n_step, window_hw[1] // n_step]

    # 仔细设置区间，可以避免采样到完全无效的区域
    y_start = 0 - window_hw[0] + window_step_hw[0]
    y_end = im_hw[0]
    x_start = 0 - window_hw[1] + window_step_hw[1]
    x_end = im_hw[1]

    for y in range(y_start, y_end, window_step_hw[0]):
        for x in range(x_start, x_end, window_step_hw[1]):
            yx_start = (y, x)
            yx_end = (y +  window_hw[0], x + window_hw[1])
            yield yx_start, yx_end


def n_step_scan_coords_gen_v2(im_hw=(512, 512), window_hw=(128, 128), n_step: float=1.):
    '''
    N步长坐标生成器，支持小数步长，可以生成滑动框左上角和右下角坐标，便于快速遍历图像
    例如窗口大小设定为[128,128]，设定为0.5步长时，代表步长为[64,64]，设定为2步长时，代表步长为[256,256]
    当前只支持生成整数坐标
    :param im_hw:       图像大小
    :param window_hw:   窗口高宽
    :param n_step:      N步长，默认为1步长
    :return: yx_start, yx_end   扫描框左上角和右下角坐标
    '''
    step_hw = [int(window_hw[0] * n_step), int(window_hw[1] * n_step)]

    # 使采样区间等于或大于边界
    y_start = 0 - window_hw[0] + step_hw[0]
    y_end = im_hw[0]
    x_start = 0 - window_hw[1] + step_hw[1]
    x_end = im_hw[1]

    for y in range(y_start, y_end, step_hw[0]):
        for x in range(x_start, x_end, step_hw[1]):
            yx_start = (y, x)
            yx_end = (y + window_hw[0], x + window_hw[1])
            yield yx_start, yx_end


if __name__ == '__main__':
    g = n_step_scan_coords_gen([512, 512], window_hw=[100, 100], n_step=1)
    for yx_start, yx_end in g:
        print(yx_start, yx_end)

    g = n_step_scan_coords_gen_v2([512, 512], window_hw=[100, 100], n_step=0.5)
    for yx_start, yx_end in g:
        print(yx_start, yx_end)

    # for ipython test speed
    # %timeit list(n_step_scan_pos_gen([51200, 51200], window_hw=[128, 128], n_step=1))
