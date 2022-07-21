import numpy as np
import cv2
import histomicstk as htk
import os
from skimage.transform import resize
from matplotlib import pylab as plt
from matplotlib.colors import ListedColormap
from histomicstk.preprocessing.color_normalization import reinhard
from histomicstk.saliency.tissue_detection import (
    get_slide_thumbnail, get_tissue_mask)
from histomicstk.preprocessing.color_normalization.\
    deconvolution_based_normalization import deconvolution_based_normalization
from tqdm import tqdm


def method_v2_no_mask():
    target_root = "./image_processing/target/TC_target/"
    source_root = "/YOUR_DIR/train_TCGA/"


    save_root = "/YOUR_SAVE_DIR/"
    target_images = os.listdir(target_root)
    source_images = os.listdir(source_root)

    index = 0
    for target_image in target_images:
        for source_image in tqdm(source_images):
            index += 1
            target_path = os.path.join(target_root,target_image)
            source_path = os.path.join(source_root,source_image)

            target = cv2.imread(target_path)
            source = cv2.imread(source_path)

            target = cv2.cvtColor(target,cv2.COLOR_BGR2RGB)
            source = cv2.cvtColor(source,cv2.COLOR_BGR2RGB)

            source_yuanshi = np.copy(source)
            stain_unmixing_routine_params = {
                'stains': ['hematoxylin', 'eosin'],
                'stain_unmixing_method': 'macenko_pca',
            }
            tissue_rgb_normalized = deconvolution_based_normalization(source, im_target=target,stain_unmixing_routine_params=stain_unmixing_routine_params)


            source_change = cv2.cvtColor(tissue_rgb_normalized, cv2.COLOR_RGB2BGR);


            target = cv2.cvtColor(target,cv2.COLOR_RGB2BGR)
            source_yuanshi = cv2.cvtColor(source_yuanshi,cv2.COLOR_RGB2BGR)

            # os.mkdir(os.path.join(save_root,str(index)))

            result_path = os.path.join(save_root,'changeA_'+ source_image)
            cv2.imwrite(result_path ,source_change)

            # save_target_path = os.path.join(save_root,'change_'+target_image)
            # cv2.imwrite(save_target_path,target)
            #
            # save_source_path = os.path.join(save_root, 'change_'+ source_image)
            # cv2.imwrite(save_source_path, source_yuanshi)



if __name__ == '__main__':
    method_v2_no_mask()
