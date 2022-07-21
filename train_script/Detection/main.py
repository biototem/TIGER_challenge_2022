import os
import torch
import torch.nn.functional as F
from DataReader.DataReader_tiger import DatasetReader
from config import device, net_in_hw, batch_size, epochs, batch_counts,\
    is_train_from_recent_checkpoint, net_save_dir, train_lr,\
    match_distance_thresh_list, process_control, data_path
import eval_utils
import uuid
import albumentations as albu
from torch.utils.data import DataLoader
import math
from tqdm import tqdm
import numpy as np
import loss_func
from lib import big_pic_result
import yaml
import cv2

get_pg_name = eval_utils.get_pg_name
get_pg_id = eval_utils.get_pg_id

save_dir = net_save_dir                                   # 模型存储路径
os.makedirs(save_dir, exist_ok=True)                      # 创建模型存储文件夹

def collate_fn(batch):
    #  batch是一个列表，其中是一个一个的元组，每个元组是dataset中_getitem__的结果
    batch = list(zip(*batch))
    image = torch.tensor(batch[0])
    mask_det = torch.tensor(batch[1])
    det_center_dict = batch[2]
    image_name = batch[3]
    del batch
    return image, mask_det,det_center_dict,image_name



def heatmap_nms(hm: np.ndarray):
    a = (hm * 255).astype(np.int32)
    a1 = cv2.blur(hm, (3, 3)).astype(np.int32)
    a2 = cv2.blur(hm, (5, 5)).astype(np.int32)
    a3 = cv2.blur(hm, (7, 7)).astype(np.int32)

    ohb = (hm > 0.).astype(np.float32)

    h = a + a1 + a2 + a3

    h = (h / 4).astype(np.float32)

    ht = torch.tensor(h)[None, None, ...]
    htm = torch.nn.functional.max_pool2d(ht, 9, stride=1, padding=4)
    hmax = htm[0, 0, ...].numpy()

    h = (h >= hmax).astype(np.float32)

    h = h * ohb

    h = cv2.dilate(h, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    return h



def main(NetClass):
    model_id = NetClass.model_id
    single_id = str(uuid.uuid1())
    print('single_id:', single_id)

#  ============================== 定义存储文件路径 =====================================

    ck_name         = '{}/{}_model.pt'          .format(save_dir, model_id)
    ck_best_name    = '{}/{}_model_best.pt'     .format(save_dir, model_id)

    ck_optim_name   = '{}/{}_optim.pt'          .format(save_dir, model_id)
    ck_restart_name = '{}/{}_restart.yml'       .format(save_dir, model_id)

    score_det_name      = '{}/{}_score_det.txt' .format(save_dir, model_id)
    score_det_2_name      = '{}/{}_score_det_2.txt' .format(save_dir, model_id)

    score_det_best_name = '{}/{}_score_det_best.txt'.format(save_dir, model_id)
    score_det_2_best_name = '{}/{}_score_det_2_best.txt'.format(save_dir, model_id)



# ===========================    定义数据增强的方式     =======================================
    transform = albu.Compose([  # 后面再仔细的调整
        albu.RandomCrop(64, 64, always_apply=True, p=1),
        albu.RandomRotate90(always_apply=False, p=0.5),  # 将输入随机旋转90度，零次或多次。
        albu.VerticalFlip(always_apply=False, p=0.5),
        albu.HorizontalFlip(always_apply=False, p=0.5),
        albu.Rotate(limit=90, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),
        albu.RandomBrightnessContrast(p=0.2),  # 随机明亮对比度
        albu.HueSaturationValue(p=0.2),
        albu.GaussNoise(p=0.5),

    ])


    # 验证集不需要进行增强
    transform_valid = albu.Compose([
    ])

    train_dataset = DatasetReader(data_path,type = "train",transforms = transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,collate_fn=collate_fn)

    valid_dataset = DatasetReader(data_path,type = "valid",transforms = transform_valid)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False, num_workers=0,collate_fn=collate_fn)


    net = NetClass()
    net = net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), train_lr, eps=1e-8, weight_decay=1e-6)
    optim_adjust = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, 1e-6)

    start_epoch = 0
    pg_max_valid_value = [-math.inf for _ in range(len(process_control)+1)]
    pg_min_loss_value = [math.inf for _ in range(len(process_control)+1)]



# ============================== 进行训练 阶段的切换  ==========================================
    if is_train_from_recent_checkpoint and os.path.isfile(ck_restart_name):
        d = yaml.safe_load(open(ck_restart_name, 'r'))
        start_epoch = d['start_epoch']
        pg_max_valid_value = d['pg_max_valid_value']
        pg_min_loss_value = d['pg_min_loss_value']

        if start_epoch not in process_control:
            new_ck_name = get_pg_name(ck_name, start_epoch, process_control)
            new_ck_optim_name = get_pg_name(ck_optim_name, start_epoch, process_control)
            net.load_state_dict(torch.load(new_ck_name, 'cpu'))
            optimizer.load_state_dict(torch.load(new_ck_optim_name, 'cpu'))




    for epoch in range(start_epoch,epochs):
        print("epoch",epoch)
        print("learning—rate",optimizer.state_dict()['param_groups'][0]['lr'])

        train_det_acc = 0
        det_acc = 0
        train_loss = 0

        if epoch in process_control:     # 需要进行阶段的变更了
            new_ck_name = get_pg_name(ck_best_name,epoch-1,process_control)       # 获取上一个阶段最好的模型参数的路径
            net.load_state_dict(torch.load(new_ck_name,'cpu'))                # 进行模型参数的权重加载
            new_ck_optim_name = get_pg_name(ck_optim_name, epoch - 1, process_control)  # 获取到上一个epoch的优化器的参数
            optimizer.load_state_dict(torch.load(new_ck_optim_name, 'cpu'))  # 进行优化器参数的加载


        pg_id = get_pg_id(epoch, process_control)
        print("pg_id",pg_id)


        net.train()    # 开启训练模式
        if pg_id == 1:      #  粗检测阶段
            net.enabled_b2_branch = False
            net.set_freeze_seg1(False)     # 开启粗检测分支
            net.set_freeze_seg2(True)      # 冻结假阳性抑制分支

        elif pg_id == 2:                    #  假阳性抑制阶段
            net.enabled_b2_branch = True
            net.set_freeze_seg1(True)        # 冻结粗检测分支
            net.set_freeze_seg2(False)       # 开启假阳性抑制分支

        elif pg_id == 3:                     # 联合训练
            net.enabled_b2_branch = True
            net.set_freeze_seg1(False)      # 开启粗检测分支
            net.set_freeze_seg2(False)      # 开启假阳性抑制分支
        else:
            raise AssertionError()

        for batch_count in tqdm(range(batch_counts)):
            for index,(image, mask_det, annotation_center, image_name) in enumerate(train_loader):
                image, mask_det = image.to(device), mask_det.to(device)
                batch_pred_det, batch_pred_det_2 = net(image)

                # 训练时检查像素准确率，这里的准确率按照类别平均检查
                with torch.no_grad():
                    batch_pred_det_tmp = batch_pred_det
                    batch_pred_det_tmp = batch_pred_det_tmp.softmax(1)[:, 1:]
                    batch_pred_det_tmp_bin = (batch_pred_det_tmp > 0.5).type(torch.float32)
                    det_acc = (batch_pred_det_tmp_bin * mask_det).sum() / (mask_det.sum() + 1e-8).item()

                    if pg_id > 1:
                        batch_pred_det_2_tmp = batch_pred_det_2
                        batch_pred_det_2_tmp = batch_pred_det_2_tmp.softmax(1)[:, 1:]

                        batch_pred_det_2_tmp_bin = ((batch_pred_det_2_tmp * batch_pred_det_tmp) > 0.5).type(torch.float32)
                        det_2_acc = ((batch_pred_det_2_tmp_bin * mask_det).sum(dim=(0, 2, 3)) / (
                                    mask_det.sum(dim=(0, 2, 3)) + 1e-8)).mean().item()
                    else:
                        det_2_acc = 0.

                if pg_id == 1:
                    _tmp = batch_pred_det.softmax(dim=1)
                    loss = loss_func.det_loss(_tmp, mask_det)

                elif pg_id == 2:
                    _tmp = batch_pred_det.softmax(dim=1)[:, 1:2]
                    _temp_2 = batch_pred_det_2.softmax(dim=1)
                    loss = loss_func.det_false_positive_loss(_tmp,_temp_2, mask_det,0.5)

                elif pg_id == 3:
                    _tmp = batch_pred_det.softmax(dim=1)
                    loss1 = loss_func.det_loss(_tmp, mask_det)

                    _temp_2 = batch_pred_det_2.softmax(dim=1)
                    loss2 = loss_func.det_false_positive_loss(_tmp[:, 1:2],_temp_2, mask_det,0.5)

                    loss = loss1 + loss2
                else:
                    raise AssertionError('Unknow process_step')
                train_det_acc += det_acc
                det_2_acc += det_2_acc
                train_loss += loss.item()

                print(
                    'epoch: {} count: {} train det acc: {:.3f} train det_2_acc: {:.3f} loss: {:.3f}'.format(epoch,batch_count,det_acc,det_2_acc,loss.item()))
                optimizer.zero_grad()
                assert not np.isnan(loss.item()), 'Found loss Nan!'
                loss.backward()
                optimizer.step()

            train_det_acc = train_det_acc / (batch_count + 1)
            det_2_acc = det_2_acc / (batch_count + 1)
            train_loss = train_loss / (batch_count + 1)

        optim_adjust.step(epoch)

        if (epoch + 1) % 1 == 0:
            with torch.no_grad():
                net.enabled_b2_branch = True
                net.eval()
                det_score_table = {'epoch': epoch,
                                   'det_pix_pred_found': 0,
                                   'det_pix_pred_fakefound': 0,
                                   'det_pix_label_found': 0,
                                   'det_pix_label_nofound': 0,
                                   'det_pix_recall': 0,
                                   'det_pix_prec': 0,
                                   'det_pix_f1': 0,
                                   'det_pix_f2': 0,
                                   }

                for dt in match_distance_thresh_list:
                    det_score_table[dt] = {
                        'found_pred': 0,  # 所有的假阳性
                        'fakefound_pred': 0,  # 所有的假阴性
                        'found_label': 0,  # 所有找到的标签
                        'nofound_label': 0,  # 所有找到的预测
                        'label_repeat': 0,  # 对应了多个pred的标签
                        'pred_repeat': 0,  # 对应了多个label的预测
                        'f1': None,
                        'recall': None,
                        'prec': None,
                    }

                det_2_score_table = {'epoch': epoch,
                                     'det_pix_pred_found': 0,
                                     'det_pix_pred_fakefound': 0,
                                     'det_pix_label_found': 0,
                                     'det_pix_label_nofound': 0,
                                     'det_pix_recall': 0,
                                     'det_pix_prec': 0,
                                     'det_pix_f1': 0,
                                     'det_pix_f2': 0,
                                     }

                for dt in match_distance_thresh_list:
                    det_2_score_table[dt] = {
                        'found_pred': 0,  # 所有的假阳性
                        'fakefound_pred': 0,  # 所有的假阴性
                        'found_label': 0,  # 所有找到的标签
                        'nofound_label': 0,  # 所有找到的预测
                        'label_repeat': 0,  # 对应了多个pred的标签
                        'pred_repeat': 0,  # 对应了多个label的预测
                        'f1': None,
                        'recall': None,
                        'prec': None,
                    }

                for valid_index, (valid_image, valid_mask_det_label, valid_annotation_center, valid_image_name) in tqdm(
                        enumerate(valid_loader)):
                    # ================================== 不知道为什么这里获得的都是元组，所以为了取数据【0】 =======================

                    valid_annotation_center = valid_annotation_center[0]

                    valid_image = valid_image[0]
                    valid_image = valid_image.cpu().detach().numpy()[:, :, :]
                    valid_image = np.transpose(valid_image, (1, 2, 0)).astype(np.float32)

                    valid_mask_det_label = valid_mask_det_label[0]
                    valid_mask_det_label = valid_mask_det_label.cpu().detach().numpy()[:, :, :]
                    valid_mask_det_label = np.transpose(valid_mask_det_label, (1, 2, 0)).astype(np.float32)

                    valid_image_name = valid_image_name

                    # ===================================================================================================================================

                    label_det_pts_temp = np.array(valid_annotation_center)
                    label_det_pts_a = []
                    for (x, y) in label_det_pts_temp:
                        label_det_pts_a.append([y, x])

                    label_det_pts = np.array(label_det_pts_a)

                    valid_image = valid_image * 255  # DataReader 里面除了255
                    wim = big_pic_result.BigPicPatch(1 + 1, [valid_image], (0, 0), net_in_hw, (0, 0), 0, 0,
                                                     custom_patch_merge_pipe=eval_utils.patch_merge_func,
                                                     patch_border_pad_value=255)
                    gen = wim.batch_get_im_patch_gen(batch_size * 3)

                    for batch_info, batch_patch in gen:
                        batch_patch = torch.tensor(np.asarray(batch_patch), dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255.
                        valid_batch_pred_det, valid_batch_pred_det2 = net(batch_patch)

                        valid_batch_pred_det = valid_batch_pred_det.softmax(1)[:, 1:]  # 粗检测预测通道
                        valid_batch_pred_det2 = valid_batch_pred_det2.softmax(1)[:, 1:]  # 假阳性抑制检测预测通道

                        valid_batch_pred = torch.cat([valid_batch_pred_det, valid_batch_pred_det2], 1)
                        valid_batch_pred = valid_batch_pred.permute(0, 2, 3, 1).cpu().detach().numpy()
                        wim.batch_update_result(batch_info, valid_batch_pred)

                    pred_pm = wim.multi_scale_result[0].data / np.clip(wim.multi_scale_mask[0].data, 1e-8, None)
                    pred_det_pm, pred_det2_pm = np.split(pred_pm, [1], -1)

                    pred_det_pm_nms = np.copy(pred_det_pm)
                    pred_det_pm_nms[:, :, 0] = pred_det_pm_nms[:, :, 0] * heatmap_nms(pred_det_pm_nms[:, :, 0])

                    if pg_id == 1:
                        pred_det2_pm_final = None
                    else:
                        pred_det2_pm_final = pred_det2_pm * pred_det_pm


                    if pg_id > 1:
                        pred_det2_pm_final_nms = np.copy(pred_det2_pm_final)
                        pred_det2_pm_final_nms[:, :, 0] = pred_det2_pm_final_nms[:, :, 0] * heatmap_nms(pred_det2_pm_final_nms[:, :, 0])


                    # 统计像素分类数据
                    if pg_id > 1:
                        det_pix_pred_bin = (pred_det2_pm_final[..., 0:1] > 0.5).astype(dtype=np.float32)
                    else:
                        det_pix_pred_bin = (pred_det_pm[..., 0:1] > 0.5).astype(dtype=np.float32)

                    det_pix_label_bin = (valid_mask_det_label[..., 0:1] > 0.5).astype(dtype=np.float32)

                    det_score_table['det_pix_pred_found'] += float((det_pix_pred_bin * det_pix_label_bin).sum(dtype=np.float32))
                    det_score_table['det_pix_pred_fakefound'] += float((det_pix_pred_bin * (1 - det_pix_label_bin)).sum(dtype=np.float32))
                    det_score_table['det_pix_label_found'] += det_score_table['det_pix_pred_found']
                    det_score_table['det_pix_label_nofound'] += float(((1 - det_pix_pred_bin) * det_pix_label_bin).sum(dtype=np.float32))

                    # =======================================  粗检测 ===========================================================================

                    det_info = eval_utils.calc_a_sample_info_points_each_class(pred_det_pm_nms, [label_det_pts,[0] * len(label_det_pts)],[0], match_distance_thresh_list,use_post_pro=False, use_single_pair=True)

                    for dt in match_distance_thresh_list:
                        for cls in [0]:
                            det_score_table[dt]['found_pred'] += det_info[cls][dt]['found_pred']
                            det_score_table[dt]['fakefound_pred'] += det_info[cls][dt]['fakefound_pred']
                            det_score_table[dt]['found_label'] += det_info[cls][dt]['found_label']
                            det_score_table[dt]['nofound_label'] += det_info[cls][dt]['nofound_label']
                            det_score_table[dt]['pred_repeat'] += det_info[cls][dt]['pred_repeat']
                            det_score_table[dt]['label_repeat'] += det_info[cls][dt]['label_repeat']

                    if pg_id > 1:
                        det_2_info = eval_utils.calc_a_sample_info_points_each_class(pred_det2_pm_final_nms, [label_det_pts,[0] * len(label_det_pts)],[0], match_distance_thresh_list,use_post_pro=False,use_single_pair=True)
                        for dt in match_distance_thresh_list:
                            for cls in [0]:
                                det_2_score_table[dt]['found_pred'] += det_2_info[cls][dt]['found_pred']
                                det_2_score_table[dt]['fakefound_pred'] += det_2_info[cls][dt]['fakefound_pred']
                                det_2_score_table[dt]['found_label'] += det_2_info[cls][dt]['found_label']
                                det_2_score_table[dt]['nofound_label'] += det_2_info[cls][dt]['nofound_label']
                                det_2_score_table[dt]['pred_repeat'] += det_2_info[cls][dt]['pred_repeat']
                                det_2_score_table[dt]['label_repeat'] += det_2_info[cls][dt]['label_repeat']

                # 计算det的像素的F1，精确率，召回率
                det_score_table['det_pix_prec'] = det_score_table['det_pix_pred_found'] / (
                            det_score_table['det_pix_pred_fakefound'] + det_score_table['det_pix_pred_found'] + 1e-8)
                det_score_table['det_pix_recall'] = det_score_table['det_pix_label_found'] / (
                            det_score_table['det_pix_label_nofound'] + det_score_table['det_pix_label_found'] + 1e-8)
                det_score_table['det_pix_f1'] = 2 * det_score_table['det_pix_prec'] * det_score_table[
                    'det_pix_recall'] / (det_score_table['det_pix_prec'] + det_score_table['det_pix_recall'] + 1e-8)
                det_score_table['det_pix_f2'] = 5 * det_score_table['det_pix_prec'] * det_score_table[
                    'det_pix_recall'] / (det_score_table['det_pix_prec'] * 4 + det_score_table['det_pix_recall'] + 1e-8)

                # 计算det的F1，精确率，召回率
                for dt in match_distance_thresh_list:
                    prec = det_score_table[dt]['found_pred'] / (
                                det_score_table[dt]['found_pred'] + det_score_table[dt]['fakefound_pred'] + 1e-8)
                    recall = det_score_table[dt]['found_label'] / (
                                det_score_table[dt]['found_label'] + det_score_table[dt]['nofound_label'] + 1e-8)
                    f1 = 2 * (prec * recall) / (prec + recall + 1e-8)
                    det_score_table[dt]['prec'] = prec
                    det_score_table[dt]['recall'] = recall
                    det_score_table[dt]['f1'] = f1

                # ===============================================  假阳性抑制 ================================================================

                # 计算pred_det2_pm_final 的F1，精确率，召回率
                for dt in match_distance_thresh_list:
                    prec_2 = det_2_score_table[dt]['found_pred'] / (
                                det_2_score_table[dt]['found_pred'] + det_2_score_table[dt]['fakefound_pred'] + 1e-8)
                    recall_2 = det_2_score_table[dt]['found_label'] / (
                                det_2_score_table[dt]['found_label'] + det_2_score_table[dt]['nofound_label'] + 1e-8)
                    f1_2 = 2 * (prec_2 * recall_2) / (prec_2 + recall_2 + 1e-8)
                    det_2_score_table[dt]['prec'] = prec_2
                    det_2_score_table[dt]['recall'] = recall_2
                    det_2_score_table[dt]['f1'] = f1_2

                out_line = yaml.safe_dump(det_score_table) + '\n' + yaml.safe_dump(det_2_score_table)

                print('epoch {}'.format(epoch), out_line)

                if pg_id == 1:
                    current_valid_value = det_score_table[match_distance_thresh_list[0]]['f1']
                elif pg_id == 2:
                    current_valid_value = det_2_score_table[match_distance_thresh_list[0]]['f1']
                elif pg_id == 3:
                    current_valid_value = det_2_score_table[match_distance_thresh_list[0]]['f1']

                # 保存最好的
                if current_valid_value > pg_max_valid_value[pg_id - 1]:
                    pg_max_valid_value[pg_id - 1] = current_valid_value
                    new_ck_best_name = get_pg_name(ck_best_name, epoch, process_control)
                    torch.save(net.state_dict(), new_ck_best_name)
                    new_score_det_best_name = get_pg_name(score_det_best_name, epoch, process_control)
                    new_score_det_2_best_name = get_pg_name(score_det_2_best_name, epoch, process_control)
                    yaml.safe_dump(det_score_table, open(new_score_det_best_name, 'w'))
                    yaml.safe_dump(det_2_score_table, open(new_score_det_2_best_name, 'w'))

                new_ck_name = get_pg_name(ck_name, epoch, process_control)
                new_ck_optim_name = get_pg_name(ck_optim_name, epoch, process_control)
                torch.save(net.state_dict(), new_ck_name)
                torch.save(optimizer.state_dict(), new_ck_optim_name)
                d = {'start_epoch': epoch + 1, 'pg_max_valid_value': pg_max_valid_value,
                     'pg_min_loss_value': pg_min_loss_value}
                yaml.safe_dump(d, open(ck_restart_name, 'w'))
                new_score_det_name = get_pg_name(score_det_name, epoch, process_control)
                new_score_det_2_name = get_pg_name(score_det_2_name, epoch, process_control)
                yaml.safe_dump(det_score_table, open(new_score_det_name, 'w'))
                yaml.safe_dump(det_2_score_table, open(new_score_det_2_name, 'w'))

if __name__ == '__main__':
    from model.model_tiger import MainNet
    main(MainNet)
