import albumentations as A


def trans_transfer(cfg: dict):
    if cfg['level'] == 'strong':
        tf = strong()
    elif cfg['level'] == 'weak':
        tf = weak()
    elif cfg['level'] == 'off':
        # If off, no transfer added
        return A.Compose([])
    else:
        raise ValueError(f'Transfer-level not legal')
    if cfg['elastic'] == 'on':
        tf += elastic()
    if cfg['cut'] == 'on':
        tf += cut()
    return A.Compose(tf)
    # return A.Compose(elastic())


def weak():
    return [
        A.OneOf([
            A.RandomGamma(gamma_limit=(75, 135), p=1),
            # 只做 hue 增强
            A.HueSaturationValue(
                hue_shift_limit=45,
                sat_shift_limit=30,
                val_shift_limit=15,
                p=1
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=1
            ),
        ], p=0.8),
        A.OneOf([
            # A.GaussianBlur(blur_limit=(3, 9), p=1),
            A.GaussNoise()
            # A.IAAAdditiveGaussianNoise(loc=10, scale=(0, 15), p=1)
        ], p=0.8),
    ]


def strong():
    # 各种增强瞎 JB 上
    return [
        A.RandomGamma(gamma_limit=(75, 135), p=0.5),
        A.RandomContrast(p=0.5),
        A.Equalize(p=0.5),
        A.Solarize(p=0.5),
        A.HueSaturationValue(
            hue_shift_limit=45,
            sat_shift_limit=30,
            val_shift_limit=15,
            p=0.5,
        ),
        A.Posterize(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.5,
        ),
        # A.Sharpen(p=0.5),
        A.GaussNoise(p=0.5),
        A.IAAAdditiveGaussianNoise(loc=10, scale=(0, 15), p=0.5),
    ]


# 关于 弹性变换：
# 下式，alpha 决定了弹性变换的变形程度
# sigma 决定了变形后高斯平滑的规模
# alpha_affine 决定了弹性变换伴随仿射变换的取值范围
# 需要注意的是，弹性变换本身不会引起太多的信息丢失（局部曲边丢失）
# 但伴随仿射变换却可能引起严重的信息丢失（整体斜边丢失）
# 此外，伴随仿射变换本身也不符合医疗图像的特征
# 而弹性变换本身则可由人体器官的延展性进行解释
# 因此， alpha_affine 总是应该置0
def elastic():
    return [A.ElasticTransform(
        alpha=500,
        sigma=50,
        alpha_affine=0,
        p=0.8,
    )]


# 关于 随机丢弃：
# 随机丢弃将破坏图片中的信息，随机将一小块区域丢弃掉，使之全黑
# 此处给出两种丢弃方式：
# 网格丢弃：规律的丢弃大量网格点附近的图块，这些图块非常小，以至于几乎不影响全局评价，因此不改变 label
# 图块丢弃：随机的丢弃某一大片图块，这些图块非常大，以至于内部的信息根本无法还原，因此改变 label
def cut():
    return [
        A.OneOf([
            # 网格尺寸小，不改变 mask
            A.GridDropout(
                # mask_fill_value=0,
                unit_size_min=2,
                unit_size_max=5,
                p=1,
            ),
            # 随机尺寸大，改变mask
            A.CoarseDropout(
                mask_fill_value=0,
                min_holes=1,
                max_holes=3,
                min_width=50,
                max_width=200,
                min_height=50,
                max_height=200,
                p=1,
            ),
        ], p=0.2),
    ]


if __name__ == '__main__':
    import numpy as np
    from utils import PPlot
    img = np.ones(shape=(100, 100, 3), dtype=np.float32) * 0.3
    lbl = np.zeros(shape=(100, 100, 7), dtype=np.uint8)
    lbl[:, :, 1] = 1
    img[50, :, :] = 1
    img[:, 50, :] = 1
    lbl[50, :, :] = 0
    lbl[:, 50, :] = 0

    pplt = PPlot()
    pplt.add(img, lbl[:, :, 0], lbl[:, :, 1])
    for t in [
        A.GridDropout(
            mask_fill_value=1,
            p=1,
        ),
        A.CoarseDropout(
            mask_fill_value=1,
            p=1,
        ),
        A.ChannelDropout(
            p=1,
        ),
    ]:
        print(t)
        r = t(image=img, mask=lbl)
        im = r['image']
        lb = r['mask']
        pplt.add(im, lb[:, :, 0], lb[:, :, 1])
    pplt.show()
