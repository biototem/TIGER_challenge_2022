import os.path

from basic import config
from utils import Timer
from process import *


def main():
    with Timer() as T:
        # 预构建流程，用于解析数据源，通常解析一遍即可、但开发初期可能频繁修改解析方案。
        if config['process.force_build'] or not os.path.exists(config['source.lib']):
            do_build(T=T)
        # 训练流程，我们可以设计许多不同的训练流程，但只选择其中一种来运行。
        do_train(T)
        # # 我们也可以将训练和验证分成两个流程
        # # do_train_save_epoch()
        # # do_valid()
        # # 测试流程，通常指定某个模型来预测和评估
        # model_path1 = join(f'~/output/{config["output.target"]}/best_dice.pth')
        # model_path2 = join(f'~/output/{config["output.target"]}/best_ce.pth')
        # do_test(model_path1, model_path2)


if __name__ == '__main__':
    main()
