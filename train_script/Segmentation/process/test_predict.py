import torch

from basic import join, names_lib
from dataset.dataset import TestDataSet
from predict import PredictManager


def do_test_predict():
    with PredictManager() as P:
        # for names in ['test']:
        for names in ['test', 'valid', 'train']:
            root = join(f'~/output/predict/{names}')
            P.root(root=root)

            # 数据组
            names = names_lib[names]
            # names = ['H1707714 1 HE'] # 这个测某一特定数据
            # 数据源
            pset = TestDataSet(
                name_list=names,
                use_format=True,
            )

            # 预测融合图集
            merged_set = P.predict(
                # model=torch.load(join('~/output/model.pth')),
                model=torch.load(join('~/output/lcex(a=0_04)-rand(密集阵)-epoch10.pth')),
                predict_set=pset,
                visual=True
            )

            # 实例图集
            instance_set = P.instance(merged_set=merged_set, visual=True)

            # 轮廓集
            contours_set = P.contours(instance_set=instance_set)

            # 信息数据集
            info_set = P.info(contours_set=contours_set, merged_set=merged_set, log=True)

            # 预测结果输出 -> 输出 xml
            P.output(info_set=info_set)

            # 预测结果评估 -> 输出评分表，label-predict
            P.evaluate_old(merged_set=merged_set, pset=pset, visual=True)
            P.evaluate(info_set=info_set, pset=pset, visual=True)

            # 预测结果可视化 -> 输出 predict, image-predict
            P.visual(info_set=info_set, pset=pset)
