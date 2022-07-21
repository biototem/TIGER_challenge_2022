import os
import torch
import torch.nn.functional as F
import numpy as np
import yaml
import imageio
import copy
from DataReader.DataReader_tiger import DatasetReader
from config import device, net_in_hw, batch_size, net_save_dir, match_distance_thresh_list, process_control, \
    eval_which_checkpoint,data_path
import albumentations as albu
from lib import big_pic_result
from torch.utils.data import DataLoader
import eval_utils
from tqdm import tqdm
from eval_utils import calc_a_sample_info_points_each_class
import cv2

# =======================================  一些配置性变量数据 =====================================
use_heatmap_nms = True
use_single_pair = True

pred_heatmap_thresh_list = (0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2)

pred_train_out_dir = 'pred_circle_no_Gaosi/cla_pred_circle_train_out_dir'
pred_valid_out_dir = 'pred_circle_no_Gaosi/cla_pred_circle_valid_out_dir'
pred_test_out_dir = 'pred_circle_no_Gaosi/cla_pred_circle_test_out_dir'

'''
命名注意
eval是评估
valid是验证集
test是测试集
eval可以是验证集，也可以是测试集

cm 代表类别图
pm 代表概率图
'''

get_pg_id = eval_utils.get_pg_id
get_pg_name = eval_utils.get_pg_name


def collate_fn(batch):
    #  batch是一个列表，其中是一个一个的元组，每个元组是dataset中_getitem__的结果
    batch = list(zip(*batch))
    image = torch.tensor(batch[0])
    mask_det_cla = torch.tensor(batch[1])
    det_center_dict = batch[2]
    image_name = batch[3]
    del batch
    return image, mask_det_cla,det_center_dict,image_name


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
    torch.set_grad_enabled(False)

    model_id = NetClass.model_id

# ======================= 获取权重路径 =================================
    ck_name         = '{}/{}_model.pt'          .format(net_save_dir, model_id)
    ck_best_name    = '{}/{}_model_best.pt'     .format(net_save_dir, model_id)

    start_epoch = 299
    pg_id = get_pg_id(start_epoch, process_control)

    print("pg_id", pg_id)

    # 测试时不需要进行增强
    transform_valid = albu.Compose([
    ])

    train_dataset = DatasetReader(data_path,type = "train",transforms = transform_valid)
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, num_workers=0,collate_fn=collate_fn)

    valid_dataset = DatasetReader(data_path,type = "valid",transforms = transform_valid)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False, num_workers=0,collate_fn=collate_fn)

    test_dataset = DatasetReader(data_path,type = "test",transforms = transform_valid)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0,collate_fn=collate_fn)

    # ===================  定义网络模型 =========================
    net = NetClass()
    net.enabled_b2_branch = True


    if eval_which_checkpoint == 'last':
        print('Will load last weight.')
        new_ck_name = get_pg_name(ck_name, start_epoch, process_control)
        net.load_state_dict(torch.load(new_ck_name, 'cpu'))
        print('load model success')
    elif eval_which_checkpoint == 'best':
        print('Will load best weight.')
        new_ck_name = get_pg_name(ck_best_name, start_epoch, process_control)
        # new_ck_name = '/media/totem_backup/totem/kiven/2_kiven_tiger_game/Tiger_game_final_test11/save_type3_m1_k1_bl/1_model_best_p3.pt'
        net.load_state_dict(torch.load(new_ck_name, 'cpu'))
        print('load model success')

    else:
        print('Unknow weight type. Will not load weight.')

    net = net.to(device)
    net.eval()

    # 进行预测
    for heatmap_thresh in pred_heatmap_thresh_list:
        # for did, cur_dataset in enumerate([train_dataset, valid_dataset, test_dataset]):
        #     out_dir = [pred_train_out_dir, pred_valid_out_dir, pred_test_out_dir][did]
        for did, cur_dataset in enumerate([test_dataset]):
            out_dir = [pred_test_out_dir][did]

            out_dir = '{}/{}'.format(out_dir, str(heatmap_thresh))  # 预测结果保存路径

            os.makedirs(out_dir, exist_ok=True)                     # 创建文件夹

            det_score_table = {}
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

            det_score_table_a2 = copy.deepcopy(det_score_table)     # 假阳性抑制分支的评分

            # =============================  选择 测试的数据集 =============================
            if cur_dataset == train_dataset:
                cur_dataloader = train_loader
            elif cur_dataset == valid_dataset:
                cur_dataloader = valid_loader
            elif cur_dataset == test_dataset:
                cur_dataloader = test_loader


            for index,(eval_image, eval_mask_det, eval_annotation_center, eval_image_name) in tqdm(enumerate(cur_dataloader)):
                eval_image = eval_image[0]
                eval_image = eval_image.cpu().detach().numpy()[:, :, :]
                eval_image = np.transpose(eval_image, (1, 2, 0)).astype(np.float32)

                eval_image_name = eval_image_name[0]
                mask_det_label = eval_mask_det[0]
                det_annotation_center = eval_annotation_center[0]

                image_name = str(index)

                image = eval_image * 255


                wim = big_pic_result.BigPicPatch(1 + 1, [image], (0, 0), net_in_hw, (0, 0), 0, 0, custom_patch_merge_pipe=eval_utils.patch_merge_func, patch_border_pad_value=255)
                gen = wim.batch_get_im_patch_gen(batch_size * 3)
                for batch_info, batch_patch in gen:
                    batch_patch = torch.tensor(np.asarray(batch_patch), dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255.
                    valid_batch_pred_det, valid_batch_pred_det2 = net(batch_patch)

                    valid_batch_pred_det = valid_batch_pred_det.softmax(1)[:, 1:]  # 粗检`测预测通道
                    valid_batch_pred_det2 = valid_batch_pred_det2.softmax(1)[:, 1:]  # 假阳性抑制检测预测通道

                    valid_batch_pred = torch.cat([valid_batch_pred_det, valid_batch_pred_det2], 1)
                    valid_batch_pred = valid_batch_pred.permute(0, 2, 3, 1).cpu().detach().numpy()
                    wim.batch_update_result(batch_info, valid_batch_pred)

                pred_pm = wim.multi_scale_result[0].data / np.clip(wim.multi_scale_mask[0].data, 1e-8, None)
                pred_det_pm, pred_det2_pm = np.split(pred_pm, [1], -1)

                pred_det_pm_rough = pred_det_pm
                pred_det2_pm_final = pred_det_pm * pred_det2_pm

                label_det_pts_temp = np.array(det_annotation_center)
                label_det_pts_a = []
                for (x, y) in label_det_pts_temp:
                    x = np.round(x).astype(np.int32)
                    y = np.round(y).astype(np.int32)
                    label_det_pts_a.append([y, x])
                label_det_pts = np.array(label_det_pts_a)


                # 粗检测也需要nms
                pred_det_pm_rough_nms = np.copy(pred_det_pm_rough)
                pred_det_pm_rough_nms[..., 0] = pred_det_pm_rough_nms[..., 0] * heatmap_nms(pred_det_pm_rough_nms[..., 0])
                # 获得相应的预测细胞核中心的坐标点
                pred_det_rough_pts = eval_utils.get_pts_from_hm(pred_det_pm_rough_nms, heatmap_thresh)

                # 在图片上面画上预测的点和mask标注点
                mix_pic_det_a1 = eval_utils.draw_hm_circle(image, pred_det_rough_pts, label_det_pts, 6)

                # 进行评分
                det_info = calc_a_sample_info_points_each_class([pred_det_rough_pts, [0] * len(pred_det_rough_pts)], [label_det_pts, [0] * len(label_det_pts)], [0],match_distance_thresh_list,use_post_pro=False, use_single_pair=use_single_pair)

                for dt in match_distance_thresh_list:
                    for cls in [0]:
                        det_score_table[dt]['found_pred'] += det_info[cls][dt]['found_pred']
                        det_score_table[dt]['fakefound_pred'] += det_info[cls][dt]['fakefound_pred']
                        det_score_table[dt]['found_label'] += det_info[cls][dt]['found_label']
                        det_score_table[dt]['nofound_label'] += det_info[cls][dt]['nofound_label']
                        det_score_table[dt]['pred_repeat'] += det_info[cls][dt]['pred_repeat']
                        det_score_table[dt]['label_repeat'] += det_info[cls][dt]['label_repeat']

                imageio.imwrite(os.path.join(out_dir, '{}_1_det_a1_image.png'.format(eval_image_name)), mix_pic_det_a1)
                imageio.imwrite(os.path.join(out_dir, '{}_1_det_a1_PM.png'.format(eval_image_name)),(pred_det_pm_rough * 255).astype(np.uint8))
                imageio.imwrite(os.path.join(out_dir, '{}_1_det_a1_PM_nms.png'.format(eval_image_name)),(pred_det_pm_rough_nms * 255).astype(np.uint8))
                yaml.dump(det_info, open(os.path.join(out_dir, '{}_det.txt'.format(eval_image_name)), 'w'))


# ==============================================  细检测  ==================================================
                pred_det2_pm_final_nms = np.copy(pred_det2_pm_final)

                if use_heatmap_nms:
                    pred_det2_pm_final_nms[..., 0] = pred_det2_pm_final_nms[..., 0] * heatmap_nms(pred_det2_pm_final_nms[..., 0])

                # 从细检测热图获得预测细胞核中心的坐标点
                pred_det_post_pts = eval_utils.get_pts_from_hm(pred_det2_pm_final_nms, heatmap_thresh)

                # 画图
                mix_pic_det_a2 = eval_utils.draw_hm_circle(image, pred_det_post_pts, label_det_pts, 6)

                # 评分
                det_info_a2 = calc_a_sample_info_points_each_class([pred_det_post_pts, [0] * len(pred_det_post_pts)],[label_det_pts, [0] * len(label_det_pts)], [0],match_distance_thresh_list, use_single_pair=use_single_pair)

                for dt in match_distance_thresh_list:
                    for cls in [0]:
                        det_score_table_a2[dt]['found_pred'] += det_info_a2[cls][dt]['found_pred']
                        det_score_table_a2[dt]['fakefound_pred'] += det_info_a2[cls][dt]['fakefound_pred']
                        det_score_table_a2[dt]['found_label'] += det_info_a2[cls][dt]['found_label']
                        det_score_table_a2[dt]['nofound_label'] += det_info_a2[cls][dt]['nofound_label']
                        det_score_table_a2[dt]['pred_repeat'] += det_info_a2[cls][dt]['pred_repeat']
                        det_score_table_a2[dt]['label_repeat'] += det_info_a2[cls][dt]['label_repeat']

                imageio.imwrite(os.path.join(out_dir, '{}_1_det_a2_image.png'.format(eval_image_name)), mix_pic_det_a2)
                imageio.imwrite(os.path.join(out_dir, '{}_1_det_a2_h_1before_PM.png'.format(eval_image_name)),(pred_det2_pm * 255).astype(np.uint8))
                imageio.imwrite(os.path.join(out_dir, '{}_1_det_a2_h_2after_PM.png'.format(eval_image_name)),(pred_det2_pm_final * 255).astype(np.uint8))
                imageio.imwrite(os.path.join(out_dir, '{}_1_det_a2_h_2after_PM_nms.png'.format(eval_image_name)),(pred_det2_pm_final_nms * 255).astype(np.uint8))

                yaml.dump(det_info_a2, open(os.path.join(out_dir, '{}_det_a2.txt'.format(eval_image_name)), 'w'))


            # 计算det a1 F1，精确率，召回率
            for dt in match_distance_thresh_list:
                prec = det_score_table[dt]['found_pred'] / (
                            det_score_table[dt]['found_pred'] + det_score_table[dt]['fakefound_pred'] + 1e-8)
                recall = det_score_table[dt]['found_label'] / (
                            det_score_table[dt]['found_label'] + det_score_table[dt]['nofound_label'] + 1e-8)
                f1 = 2 * (prec * recall) / (prec + recall + 1e-8)
                det_score_table[dt]['prec'] = prec
                det_score_table[dt]['recall'] = recall
                det_score_table[dt]['f1'] = f1

            yaml.dump(det_score_table, open(os.path.join(out_dir, 'all_det.txt'), 'w'))

            # 计算det a2 F1，精确率，召回率
            for dt in match_distance_thresh_list:
                prec = det_score_table_a2[dt]['found_pred'] / (
                        det_score_table_a2[dt]['found_pred'] + det_score_table_a2[dt]['fakefound_pred'] + 1e-8)
                recall = det_score_table_a2[dt]['found_label'] / (
                        det_score_table_a2[dt]['found_label'] + det_score_table_a2[dt]['nofound_label'] + 1e-8)
                f1 = 2 * (prec * recall) / (prec + recall + 1e-8)
                det_score_table_a2[dt]['prec'] = prec
                det_score_table_a2[dt]['recall'] = recall
                det_score_table_a2[dt]['f1'] = f1

            yaml.dump(det_score_table_a2, open(os.path.join(out_dir, 'all_det_2.txt'), 'w'))


if __name__ == '__main__':
    from model.model_tiger import MainNet
    main(MainNet)

    # from model_0_33.model_v7_tiger import MainNet
    # main(MainNet)