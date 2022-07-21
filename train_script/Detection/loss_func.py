#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ProjectD_xf_follow_wentai 
@File    ：loss_func.py
@IDE     ：PyCharm 
@Author  ：kiven
@Date    ：2022/3/3 11:07 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union

def det_loss(batch_pred_det: torch.Tensor, batch_label_det: torch.Tensor, det_weights: Union[torch.Tensor, str]='auto'):
    '''
    注意，这里要求 y_true 是 onehot向量，而不是类别标量
    :param batch_pred_det:
    :param batch_label_cla:
    :param det_weights:
    :return:
    '''
    batch_label_det_final = torch.cat([1-batch_label_det, batch_label_det], dim=1)   # 通道0背景，通道1是细胞核预测
    pos_n = batch_label_det_final[:, 1, :, :].sum()
    neg_n = batch_label_det_final[:, 0, :, :].sum()
    a = (neg_n / (pos_n+1))


    det_weights = torch.tensor([a], dtype=torch.float32, device=batch_label_det_final.device)
    print("det_weights",det_weights)
    batch_pred_det = torch.where(batch_pred_det > 1e-8, batch_pred_det, batch_pred_det + 1e-8)
    loss = -((batch_label_det_final[:,0,:,:] * torch.log(batch_pred_det[:,0,:,:]) + det_weights * batch_label_det_final[:,1,:,:] * torch.log(batch_pred_det[:,1,:,:]) )).sum() /  (batch_pred_det.shape[0] * batch_pred_det.shape[1] * batch_pred_det.shape[2] * batch_pred_det.shape[3])
    return loss



def det_false_positive_loss(batch_pred_det, batch_pred_det_2: torch.Tensor, batch_label_det_2: torch.Tensor, threshold=0.5):
    '''
    注意，这里要求 batch_label_cla 是 onehot 向量，而不是类别标量
    :param batch_pred_det:
    :param batch_pred_cla:
    :param batch_label_cla:
    :param cla_weights:
    :param threshold:
    :return:
    '''
    batch_label_det_2_fin = torch.cat([1 - batch_label_det_2, batch_label_det_2], dim=1)  # 通道0背景，通道1是细胞核预测
    indicator = (batch_pred_det.detach() >= threshold).type(torch.float32)
    batch_label_det_2_final = batch_label_det_2_fin * indicator

    pos_n = batch_label_det_2_final[:, 1, :, :].sum()
    neg_n = batch_label_det_2_final[:, 0, :, :].sum()

    a = neg_n / (pos_n)
    a = a + 1
    det_weights = torch.tensor([1, a], dtype=torch.float32, device=batch_label_det_2.device)
    print("cla_weight", a)
    det_weights = torch.reshape(det_weights, [1, -1, 1, 1])

    pred_final = torch.where(batch_pred_det_2 > 1e-10,batch_pred_det_2, batch_pred_det_2 + 1e-10)

    loss = -torch.sum( det_weights * indicator * batch_label_det_2_fin * torch.log(pred_final)) / (torch.sum(indicator) + 1e-10)

    return loss

