from basic import join

import os
import numpy as np
import cv2
from tqdm import tqdm
from histomicstk.preprocessing.color_normalization.deconvolution_based_normalization import deconvolution_based_normalization

from utils import PPlot


def do_convert(targets, sources, save_root):
    os.makedirs(join(save_root), exist_ok=True)
    for target in targets:
        target = cv2.imread(target)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        for source in tqdm(sources):
            name = os.path.basename(source)

            source = cv2.imread(source)

            source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)

            stain_unmixing_routine_params = {
                'stains': ['hematoxylin', 'eosin'],
                'stain_unmixing_method': 'macenko_pca',
            }
            tissue_rgb_normalized = deconvolution_based_normalization(
                source,
                im_target=target,
                stain_unmixing_routine_params=stain_unmixing_routine_params
            )

            source_change = cv2.cvtColor(tissue_rgb_normalized, cv2.COLOR_RGB2BGR)

            # target = cv2.cvtColor(target, cv2.COLOR_RGB2BGR)
            # source_yuanshi = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
            # PPlot().title('target', 'source', 'source_change', 'source_yuanshi')\
            #     .add(target, source, source_change, source_yuanshi).show()

            result_path = join(save_root, name)
            cv2.imwrite(result_path, source_change)

            # save_target_path = os.path.join(save_root,'change_'+target_image)
            # cv2.imwrite(save_target_path,target)
            #
            # save_source_path = os.path.join(save_root, 'change_'+ source_image)
            # cv2.imwrite(save_source_path, source_yuanshi)


if __name__ == '__main__':
    tgs = [
        '/media/totem_data_backup/totem/tiger-training-data/wsirois/roi-level-annotations/tissue-cells/images/100B_[35129, 6567, 36429, 7757].png',
        # '/media/totem_data_backup/totem/tiger-training-data/wsirois/roi-level-annotations/tissue-cells/images/114S_[25546, 18605, 26758, 19785].png'
    ]
    scs = [
        '/media/totem_data_backup/totem/tiger-training-data/wsirois/roi-level-annotations/tissue-bcss/images/TCGA-OL-A5RW-01Z-00-DX1.E16DE8EE-31AF-4EAF-A85F-DB3E3E2C3BFF_[2888, 3422, 3494, 3935].png',
        # '/media/totem_data_backup/totem/tiger-training-data/wsirois/roi-level-annotations/tissue-bcss/images/TCGA-LL-A73Y-01Z-00-DX1.50C20931-3AA9-40B4-8A73-56B1976423A8_[34061, 24725, 35279, 25785].png'
    ]
    do_convert(targets=tgs, sources=scs, save_root='~/resource/data/color_normalized')
