from process import do_calculate, do_evaluate
from utils import Timer


def main():
    # 用于评估的模型名称
    # model_names = ['score', 'loss']
    model_names = [f'auto-save-epoch-{epoch+1}' for epoch in range(20)]

    # # 工作目录
    # root = '~/output/v4-3'
    for root in [
        '~/output/v5-2',
    ]:

        with Timer() as T:
            do_calculate(model_names=model_names, root=root, T=T, visual=False)


        do_evaluate(root=root)


if __name__ == '__main__':
    main()
