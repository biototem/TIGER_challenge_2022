#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 15:46:37 2022

@author: biototem
"""

import numpy as np
import math
from utils.opencv_utils import OpenCV
import cv2


def get_nuclei_amount_matrix(nuclei_info, svs_width, svs_height, step):
    # 获取指定步长的细胞数量矩阵尺寸
    width_step_num = math.ceil(svs_width / step)
    height_step_num = math.ceil(svs_height / step)

    # 获取初始为0的细胞数量矩阵
    nuclei_amount_matrix = np.zeros((height_step_num, width_step_num), dtype=np.float32)

    # 将细胞核信息中每个细胞的坐标转换为指定步长的细胞数量矩阵映射坐标
    region_coord = np.array([nuclei_info[:, 1] // step, nuclei_info[:, 0] // step])

    # 将映射坐标转换为复数，Numpy.unique方法便能够根据映射坐标统计映射坐标的重复数量（细胞数量）
    region_coord_plural = region_coord[0, :] + region_coord[1, :] * 1j

    # 获取复数映射坐标及其重复数量（细胞数量）
    # Numpy.unique是向量化运算，因此能够快速统计区域细胞数量
    region_coord_plural_count = np.unique(region_coord_plural, return_counts=True)

    # 将复数映射坐标及其重复数量（细胞数量）映射到细胞数量矩阵
    loop = range(region_coord_plural_count[0].shape[0])
    x_list = list(map(lambda i: np.real(region_coord_plural_count[0][i]).astype(np.int),loop))
    y_list = list(map(lambda i:np.imag(region_coord_plural_count[0][i]).astype(np.int),loop))
    for i in loop:
        nuclei_amount_matrix[x_list[i], y_list[i]] = region_coord_plural_count[1][i]

    return nuclei_amount_matrix


def get_tumor_bulk(seg_result,
                        dimensions, level_downsamples,
                        contour_area_threshold = 0.05,
                        invasive_tumor_threshold_list = [0.05,0.95],
                        new_method = False,kernel_size=(200, 200)):

    result_to_statistics =  (seg_result==1) | (seg_result==2) | (seg_result==6)     
    ori_contours = OpenCV(result_to_statistics).find_contours(is_binary = True,is_erode_dilate = True,mode=3)
    max_contours_area = 0
    for contour in ori_contours:
        tmp_area = cv2.contourArea(contour)
        if tmp_area > max_contours_area:
            max_contours_area = tmp_area
    keep_contours = []
    for  contour in ori_contours:
        tmp_area = cv2.contourArea(contour)
        if tmp_area > max_contours_area * contour_area_threshold:
            keep_contours.append(contour)
    
    final_contours = []
    for contour in keep_contours:
        contour_mask = np.zeros(seg_result.shape,dtype= np.uint8)
        contour_mask = cv2.fillPoly(contour_mask, [contour], [1])
        contour_mask = contour_mask * result_to_statistics
        # tmp_mask = seg_result==1 
        # tmp_mask = tmp_mask & contour_mask
        invasive_tumor_rate = np.sum((seg_result==1) * contour_mask)/np.sum(contour_mask)
        if invasive_tumor_rate >= min(invasive_tumor_threshold_list) and \
            invasive_tumor_rate <= max(invasive_tumor_threshold_list):
            final_contours.append(contour)
                
    final_mask = np.zeros(seg_result.shape,dtype= np.uint8)
    final_mask = cv2.fillPoly(final_mask,final_contours, [1])
    del ori_contours,final_contours,keep_contours,result_to_statistics
    return final_mask


def get_tumor_bulk_new(seg_result,final_mask,level_downsamples,spacing,kernel_size=(200, 200)):
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    ori_tumor_result = seg_result==1
    ori_tumor_contours = OpenCV(ori_tumor_result).find_contours(is_binary = True,is_erode_dilate = False,mode=3)
    final_tumor_contours = []
    for contour in ori_tumor_contours:
        if cv2.contourArea(contour) > 30/spacing[0]*30/spacing[1]/level_downsamples/level_downsamples:
            final_tumor_contours.append(contour)
    new_tumor_result = np.zeros(seg_result.shape,dtype= np.uint8)
    new_tumor_result = cv2.fillPoly(new_tumor_result,final_tumor_contours, [1])
    # del ori_tumor_contours,final_tumor_contours
    fina_mask_before = final_mask * new_tumor_result
    # final_mask_after = OpenCV(fina_mask_before).erode_dilate(fina_mask_before, erode_iter=6, dilate_iter=9, kernel_size=(30, 30))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    final_mask_after = cv2.morphologyEx(fina_mask_before,cv2.MORPH_OPEN,kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    final_mask_after = cv2.dilate(final_mask_after, kernel, iterations=1)
    final_mask_after = final_mask_after - fina_mask_before
    final_mask_new = final_mask_after * final_mask * ((seg_result==2) | (seg_result==6))
    del fina_mask_before,final_mask_after,ori_tumor_result
    return final_mask_new


def tumor_bulk_save(seg_result,writer,dimensions,level_downsamples = 4,tile_size = 256):
    for y in range(0, dimensions[1], tile_size):
        for x in range(0, dimensions[0], tile_size):
            top = round(y/level_downsamples)
            bottom = min(dimensions[1], round(top + tile_size/level_downsamples))
            left = round(x/level_downsamples)
            right = min(dimensions[0], round(left + tile_size/level_downsamples))
            if np.sum(seg_result[top:bottom,left:right]) > 0:
                tile = OpenCV(seg_result[top:bottom,left:right]).resize(tile_size,tile_size)
                writer.write_segmentation(tile, x=x, y=y)
    

def get_score(final_mask,nuclei_det_prediction,
              dimensions,level_downsamples,step = 256):

    nuclei_amount_matrix = get_nuclei_amount_matrix(nuclei_det_prediction, dimensions[0], dimensions[1], step)
    # nuclei_amount_matrix = OpenCV(nuclei_amount_matrix).resize(final_mask.shape[1],final_mask.shape[0])
    # score = min(round(100*np.sum(nuclei_amount_matrix>10)/np.sum(final_mask)),100)
    final_mask2 = OpenCV(final_mask).resize(nuclei_amount_matrix.shape[1],nuclei_amount_matrix.shape[0])
    nuclei_amount_matrix[final_mask2==0]=0
    score = np.sum(nuclei_amount_matrix) * math.pi * 6**2 /level_downsamples /level_downsamples / np.sum(final_mask)
    del final_mask2
    return score


if __name__ == '__main__':
    import multiresolutionimageinterface as mir
    reader = mir.MultiResolutionImageReader()
    seg_mask = reader.open('your ASAP result tif filepath')
    dimensions = seg_mask.getDimensions()
    process_level = 2
    level_downsamples = round(seg_mask.getLevelDownsample(process_level))
    results = seg_mask.getUCharPatch(
                startX=0, startY=0, width = seg_mask.getLevelDimensions(process_level)[0], 
                height = seg_mask.getLevelDimensions(process_level)[1], level=process_level).squeeze()
    results = get_tumor_bulk(results,
                        dimensions, level_downsamples,
                        contour_area_threshold = 0.05,
                        invasive_tumor_threshold_list = [0.05,0.95])