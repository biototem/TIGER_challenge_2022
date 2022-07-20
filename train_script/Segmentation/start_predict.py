from process import do_predict
from utils import Timer


def main():
    # 用于评估的模型名称
    # model_names = ['score', 'loss']
    # model_name = f'auto-save-epoch-{10}'
    # 工作目录
    root = '~/output/v5-2'
    # 耗时操作，使用模型预测数据集，生成并保存原始统计数据（模型 X 在图片 Z 中 gt=A 而 pr=B 的像素数量）
    # 一并保存计算时的训练集、验证集分配情况
    for epoch in [17, 19]:
        model_name = f'auto-save-epoch-{epoch}'
        with Timer() as T:
            do_predict(model_name=model_name, root=root, T=T, visual=True)


if __name__ == '__main__':
    main()
