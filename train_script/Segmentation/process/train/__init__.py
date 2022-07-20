import torch
from torch.utils.data import DataLoader

from basic import config, source
from utils import Assert, Timer
from model import ModelConfig
from dataset import Trainset
from .helper import Helper
from .trainer import TrainEpoch


def do_train(T: Timer = None):
    Assert.not_none(source_lib=config['source.lib'], root=config['output.root'])
    Assert.file_exist(config['source.lib'])
    # 加载数据集 -> 选择目标数据集并指定模式
    if T: T.track(' -> loading trainset')
    # TODO: 临时改变:将训练集和测试集和验证集中的难样本合并一起去
    trainset = Trainset(
        names=source['group']['train'],
        # names=source['group']['dev_test'],
        return_label=True,
        shuffle=True,
        # length=42,
        cropper_rotate=config['dataset.train.rotate'],
        cropper_scaling=config['dataset.train.scaling'],
    )
    # 使用 loader
    train_loader = DataLoader(
        dataset=trainset,
        batch_size=config['train.batch_size'],
        num_workers=config["train.num_workers"]
    )

    # 加载模型
    if T: T.track(' -> initializing net')
    model = ModelConfig.net()
    # net = torch.nn.DataParallel(net)
    model.cuda()

    # 训练项
    if T: T.track(' -> initializing train-listener')
    loss = ModelConfig.loss()
    optimizer = ModelConfig.optimizer(model)
    scheduler = ModelConfig.scheduler(optimizer)

    # 现在 optimizer 和 scheduler 都将在 Epoch 中执行
    if T: T.track(' -> loading component-epoch')
    train_epoch = TrainEpoch(
        model=model,
        loss=loss,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    train_helper = Helper(model=model, T=T)

    with train_helper:
        for epoch in range(config['train.epoch']):
            if T: T.track(f' -> Epoch:{epoch + 1} with LR: {train_epoch.get_lr()}')
            train_epoch.run(train_loader, epoch)
            # 听说这个魔法可以消灭 cuda 缓存，我不确定，我试试……
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            train_helper.process(epoch, visual=False)
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
