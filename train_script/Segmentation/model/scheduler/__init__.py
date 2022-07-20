import torch.optim.lr_scheduler as lr_scheduler
import pytorch_warmup
from basic import config, InterfaceProxy


def scheduler(optimizer) -> InterfaceProxy:
    """
    该模块继承自 浩森 源码，现在集成了 warmup
    由于 scheduler 和 warmup 相关的接口参数较为复杂，相应的封装逻辑也很臃肿

    所以，这里使用了自己写的 InterfaceProxy 作为代理转换接口
    返回对象对外只暴露一个 .step 方法，该方法由 step_proxy 函数实现
    而 step_proxy 关联 sch.step 方法和 warmup.dampen 方法
    当配置文件决定不使用 scheduler 和 warmup 时，这两个方法将由 InterfaceProxy.EMPTY 代理（空函数，什么也不做）

    顺便，之所以写了一个 InterfaceProxy，就是因为这个函数、以及风格迁移模块代码的存在
    """
    sch_name = config['model.scheduler']
    if sch_name is None:
        sch = InterfaceProxy(step=InterfaceProxy.EMPTY)
    elif sch_name == 'coswarm':
        # sch = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, 2, 0)
        sch = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 2, 2, 1e-5)
    elif sch_name == 'step':
        sch = lr_scheduler.StepLR(optimizer, 10, 0.3, last_epoch=-1)
    elif sch_name == 'plateau':
        # 这里似乎有问题，mode=min表明评估函数越小越好，但这里的评估函数如果是指的metrics，显然metrics越大越好
        # 开启日志 -> verbose=True
        sch = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, threshold=0.01, min_lr=1e-6)
    elif sch_name == 'cosine':
        sch = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epoch'], eta_min=0)
    elif sch_name == 'linear':
        # # 看不懂
        # def lambda_rule(epoch):
        #     lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
        #     return lr_l
        # schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        raise NotImplementedError(f'{sch_name} not implemented')
    else:
        raise NotImplementedError(f'{sch_name} not implemented')

    wmp_name = config['model.warmup']
    if wmp_name is None:
        warmup = InterfaceProxy(dampen=InterfaceProxy.EMPTY)
    elif wmp_name == 'line':
        warmup = pytorch_warmup.LinearWarmup(optimizer, warmup_period=5)
        warmup.last_step = -1
    else:
        raise NotImplementedError(f'{wmp_name} not implemented')

    def step_proxy(epoch):
        sch.step(epoch)
        warmup.dampen()
    return InterfaceProxy(step=step_proxy)
