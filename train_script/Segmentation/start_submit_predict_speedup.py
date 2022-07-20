import gc
import os

from basic import join
from process import do_submit_predict_speedup
from utils import Timer

IMAGE_ROOT = '/YOUR_DIR/wsibulk/images'
MASK_ROOT = '/YOUR_DIR/wsibulk/tissue-masks'
MASK_SUFFIX = '_tissue'



def main():
    # 用于评估的模型名称
    # model_names = ['score', 'loss']
    model_path = join('~/output/v4-experiment/v4-van-norm-aux-12.pth')
    with Timer() as T:
        for fname in os.listdir(IMAGE_ROOT):
            name, ext = os.path.splitext(fname)
            if ext != '.tif': continue
            # 预测图目录
            image_path = join(IMAGE_ROOT, f'{name}.tif')
            mask_path = join(MASK_ROOT, f'{name}{MASK_SUFFIX}.tif')
            T.track(f' -> start predicting {name}')
            assert os.path.exists(image_path)
            assert os.path.exists(mask_path)
            do_submit_predict_speedup(
                name=name,
                model_name=model_path,
                image_path=image_path,
                mask_path=mask_path,
                T=T, visual=True
            )
            # 释放内存
            gc.collect()


def main_single():
    # 用于评估的模型名称
    # model_names = ['score', 'loss']
    model_path = join('~/output/v4-experiment/v4-van-norm-aux-12.pth')
    with Timer() as T:
        # 预测图目录
        image_path = join('~/experiment/submit/input/130S.tif')
        mask_path = join('~/experiment/submit/input/mask/130S.tif')
        # mask_path = join('~/experiment/submit/source/wsibulk/annotations-tumor-bulk/masks/130S.tif')
        # image_path = join('~/experiment/submit/source/wsibulk/images/TC_S01_P000017_C0001_B106.tif')
        # mask_path = join('~/experiment/submit/source/wsibulk/tissue-masks/TC_S01_P000017_C0001_B106_tissue.tif')

        do_submit_predict_speedup(
            name='130S',
            model_name=model_path,
            image_path=image_path,
            mask_path=mask_path,
            T=T, visual=True
        )
        # 释放内存
        gc.collect()


if __name__ == '__main__':
    main_single()
