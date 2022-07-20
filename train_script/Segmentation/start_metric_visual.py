from process import Visualizer


def main():
    # 工作目录
    root = '~/output/v5-2'
    names = [f'auto-save-epoch-{i}' for i in [10, 17, 19]]
    # names = [f'auto-save-epoch-{i + 1}' for i in range(20)]
    # 快速操作，选择性的展示评估结果
    V = Visualizer(root=root)
    V.dice(names=names)
    V.score(names=names)


if __name__ == '__main__':
    main()
