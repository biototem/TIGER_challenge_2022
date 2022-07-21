#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ProjectD_xf_follow_wentai 
@File    ：config.py
@IDE     ：PyCharm 
@Author  ：kiven
@Date    ：2022/3/2 22:02 
'''

import os

# base setting
## 项目根目录路径，一般不需要修改
project_root = os.path.split(__file__)[0]
print(project_root)

## 指定运行在哪个设备
device = 'cuda:2'
## 训练参数，分别是学习率，训练轮数，每批量大小
train_lr = 1e-3
epochs = 299
batch_size = 100

batch_counts = 1

data_path = '/YOUR_TRAINING_DATA_DIR/'

net_in_hw = (64, 64)
net_out_hw = (64, 64)

# 0到第一个数字为只训练检测，第一个数字到第二个数字为只训练分类检测，第二个数字后为同时启动训练
# process_control = [50, 100,]
process_control = [81, 135]



assert len(process_control) == 2


# 特别处理1，让分类分支也只是训练检测，不分类
make_cla_is_det = False
# 特别处理2，是否使用排斥编码
use_repel_code = True

match_distance_thresh_list = [6, 9]

# 是否从最近检查点开始训练
is_train_from_recent_checkpoint = True
# eval是否使用best检查点
eval_which_checkpoint = 'best'
assert eval_which_checkpoint in ('last', 'best', 'minloss')


# dataset setting
## 数据集位置
dataset_path = project_root + '/data_2'
#
# # net setting


# save_postfix
save_postfix = '_type3_m1_k1_bl'



# ## 指定模型保存位置和日志保存位置
net_save_dir = project_root + '/save' + save_postfix
net_train_logs_dir = project_root + '/logs' + save_postfix

