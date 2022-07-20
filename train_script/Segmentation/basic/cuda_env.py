import os
from torch.backends import cudnn


# 开启加速
cudnn.benchmark = True
# 系统环境设置
# torch.cuda.set_device(device=hyp['train.device'])
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
