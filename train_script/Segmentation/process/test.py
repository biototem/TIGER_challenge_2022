import torch

from basic import join, names_lib
from dataset.dataset import TestDataSet
from predict import PredictManager


def do_test():
    with PredictManager() as P:
        for model_name in [
            # 'auto-save-epoch-7',
            # 'auto-save-epoch-12',
            # 'auto-save-epoch-16',
            # 'auto-save-epoch-18',
            'auto-save-epoch-4',
            'auto-save-epoch-8',
            'auto-save-epoch-16',
        ]:
            # model = torch.load(join(f'~/output/lung_v4-1/latest/{model_name}.pth'))
            model = torch.load(join(f'~/output/lung_v4-2/latest/{model_name}.pth'))
            # for key in ['dev_test']:
            for key in ['test', 'valid', 'train']:
                P.root(root=join(f'~/output/predict/{model_name}/{key}'))
                evaluate_sets = {
                    # 'list[1, 2]': {},
                    # 'list[2]': {},
                    'list[1]': {},
                }

                # 本项目数据规模比肾小管大,虽然内存能勉强承下全部 sight 但很勉强, 而且意义不大
                # 因此此处分开跑
                for name in names_lib[key]:
                    for scaling in [
                        # 'list[1, 2]',
                        # 'list[2]',
                        'list[1]',
                    ]:
                        # 数据源
                        pset = TestDataSet(
                            name_list=[name],
                            use_format=True,
                            scaling=scaling,
                        )

                        # 预测融合图集
                        merged_set = P.predict(
                            model=model,
                            predict_set=pset,
                        )

                        # 预测结果可视化
                        P.visual_old(merged_set=merged_set, pset=pset, suffix=f'-{scaling}')

                        # 预测结果评估 -> 输出评分表，会破坏原数组
                        evaluate_set = P.evaluate_old(merged_set=merged_set, pset=pset, log=True, suffix=f'-{scaling}')
                        for block_name, evaluate_result in evaluate_set.items():
                            evaluate_sets[scaling][block_name] = evaluate_result

                # 评估汇总
                for scaling in [
                    # 'list[1, 2]',
                    # 'list[2]',
                    'list[1]',
                ]:
                    P.evaluate_all(evaluate_set=evaluate_sets[scaling], suffix=f'-{scaling}')
