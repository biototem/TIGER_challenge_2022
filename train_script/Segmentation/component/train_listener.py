import torch
import os
import yaml
import csv
from torch.utils.tensorboard import SummaryWriter
from segmentation_models_pytorch.utils import base

from basic import join, names_lib, hyp
from predict import PredictManager
from utils import Timer
from dataset import TestDataSet as PredictSet


class TrainListener(object):
    """
    用于训练过程中的日志记录
    """

    def __init__(self, model, T: Timer = None):
        # 指定输出目录 output/target
        self.root = join(hyp["output.root"], hyp['output.target'])
        os.makedirs(self.root, exist_ok=True)
        # 保存参数文件到本地 - 备份 yml 至 output 下，以便我们知道训练结果是用哪套参数跑出来的
        with open(join(self.root, "hyp.yaml"), mode="w", encoding="utf-8") as fp:
            yaml.dump(dict(hyp), fp)

        self.model = model
        self.conditions = {}
        self.watches = []
        self.predict_set = PredictSet(
            name_list=names_lib['test'],
            # name_list=names_lib["dev_test"],
            use_migrate=hyp['dataset.use_migrate'],
            use_confidence=hyp['dataset.use_confidence'],
        )
        self.T = T or Timer()

    def watch(self, metric):
        """
        新增日志监听项，key是loss或metrics的名字
        """
        self.watches.append(metric)
        return self

    def listen(self, name: str, handle: str):
        """
        新增保存模型的条件监听，name是loss或metrics的名字
        handle是最大或最小（一般的，loss最小化，metrics最大化）
        条件保存在 conditions 中, v 存放的是最优值, f 存放的是最优判定函数
        """
        if handle == 'maximum':
            self.conditions[name] = {
                'v': 0,
                'f': lambda v: v > self.conditions[name]['v']
            }
        elif handle == 'minimum':
            self.conditions[name] = {
                'v': 1e9,
                'f': lambda v: v < self.conditions[name]['v']
            }
        else:
            raise Exception('handle must be "maximum" or "minimum"')
        return self

    def update(self, valid_logs, epoch: int):
        """
        检查训练效果，维护最优模型的更新
        """
        self.T.track(' -> try updating model weights')
        T = self.T.tab()
        flag = False
        for name in self.conditions.keys():
            v = valid_logs[name]
            if not self.conditions[name]['f'](v):
                T.track(f' -> condition: {name} failed')
                continue
            T.track(f' -> condition: {name} success, saving to {self.root}')
            self.conditions[name]['v'] = v
            torch.save(self.model, join(self.root, '%s.pth' % name))
            with open(join(self.root, '%s.txt' % name), 'w') as f:
                f.write('epoch: %d\n' % (epoch+1))
                f.write('{}: {}\n'.format(name, valid_logs[name]))
                for w in self.watches:
                    f.write(' -> {}: {}\n'.format(w.__name__, valid_logs[w.__name__]))
                    # 像素级加更
                    if isinstance(w, base.Metric):
                        f.write(' -> {}: {}\n'.format(f'{w.__name__}_pix', valid_logs[f'{w.__name__}_pix']))
            flag = True
        # T.track(' -> updating latest weights')
        # torch.save(self.model, join(self.root, hyp['output.model']))
        if (epoch + 1) % hyp['train.test_epoch'] == 0:
            T.track(f' -> updating epoch {epoch + 1} weights')
            torch.save(self.model, join(self.root, f'auto-save-epoch-{epoch+1}.pth'))
            # T.track(f' -> do testing')
            # self.test_visual(target=f'epoch-{epoch+1}')
        return flag

    def __enter__(self):
        self.logger = SummaryWriter(
            log_dir=join(f'~/cache/tensorboard/{hyp["output.target"]}'),
            comment=hyp['output.target']
        )
        self.log_file = open(join(self.root, hyp['output.log']), 'w')
        self.writer = csv.writer(self.log_file, delimiter=';')
        train_header = map(lambda w: 'train/%s' % w.__name__, self.watches)
        valid_header = map(lambda w: 'valid/%s' % w.__name__, self.watches)
        valid_pix_header = ('valid/%s_pix' % w.__name__ for w in self.watches if isinstance(w, base.Metric))
        self.writer.writerow(tuple(train_header) + tuple(valid_header) + tuple(valid_pix_header))

    def log(self, epoch: int, train_logs, valid_logs):
        # self.log_file.write('epoch: %d\n' % epoch)
        for w in self.watches:
            self.logger.add_scalar("%s/Train" % w.__name__, train_logs[w.__name__], epoch + 1)
            self.logger.add_scalar("%s/Valid" % w.__name__, valid_logs[w.__name__], epoch + 1)
            if isinstance(w, base.Metric):
                self.logger.add_scalar("%s_pix/Valid" % w.__name__, valid_logs[f'{w.__name__}_pix'], epoch + 1)
            # self.log_file.write(' -> {}: {}\n'.format(name, valid_logs[name]))
        # self.log_file.write(' .. \n')
        train_ws = map(lambda w1: '%.2f' % train_logs[w1.__name__], self.watches)
        valid_ws = map(lambda w1: '%.2f' % valid_logs[w1.__name__], self.watches)
        valid_pix_ws = ('%.2f' % valid_logs[f'{w1.__name__}_pix'] for w1 in self.watches if isinstance(w1, base.Metric))
        self.writer.writerow(tuple(train_ws) + tuple(valid_ws) + tuple(valid_pix_ws))
        self.log_file.flush()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.T.track(f' -> ending training')
        self.logger.close()
        self.log_file.close()
        return False

    def test_visual(self, target: str = None):
        """
        生成测试结果图
        """
        print('making predictions!')
        with PredictManager(T=self.T.tab()) as P:
            # 指定目录
            if target:
                P.root(join(self.root, target))
            else:
                P.root(self.root)
            # 预测融合图
            merged_set = P.predict(self.model, self.predict_set)
            # 预测结果可视化
            P.visual_old(merged_set=merged_set, pset=self.predict_set)
            # 评估
            evaluate_set = P.evaluate_old(merged_set, self.predict_set, log=True)
            # 评估汇总
            P.evaluate_all(evaluate_set=evaluate_set)


def test():
    model = torch.load(join(f'~/output/latest/model.pth'))
    predict_set = PredictSet(
        # name_list=names_lib['test'],
        name_list=names_lib["dev_test"],
        use_migrate=hyp['dataset.use_migrate'],
        use_confidence=hyp['dataset.use_confidence'],
    )
    with PredictManager() as P:
        # 指定目录
        P.root(join('~/output/latest/test_visual'))
        merged_set = P.predict(model, predict_set, visual=True)
        P.evaluate_old(merged_set, predict_set, log=True, visual=True)


if __name__ == '__main__':
    test()
