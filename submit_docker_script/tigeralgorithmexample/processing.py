from typing import Tuple, Any
# import math
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import torch
# import cv2
from tqdm import tqdm
# from nuclei_det.det_processing import process_image_prediction_2, process_image_get_pts
from nuclei_det.det_processing_loader_merge import loader_prediction
# from nuclei_det.new_det_processing import do_process_big_patch
from utils.tils_score_calc import get_tumor_bulk,get_score#,get_tumor_bulk_new
# from utils.delaunay_baseline import get_cancer_region_baseline
from PIL import Image
Image.MAX_IMAGE_PIXELS=700000000


from gcio import (
    TMP_DETECTION_OUTPUT_PATH,
    TMP_SEGMENTATION_OUTPUT_PATH,
    TMP_TILS_SCORE_PATH,
    copy_data_to_output_folders,
    get_image_path_from_input_folder,
    get_tissue_mask_path_from_input_folder,
    initialize_output_folders,
)
from rw import (
    READING_LEVEL,
    WRITING_TILE_SIZE,
    NUCLEI_DET_MODEL_INPUT_SIZE,
    NUCLEI_DET_BIG_TILE_SIZE,
    NUCLEI_DET_TILE_STEP,
    DetectionWriter,
    SegmentationWriter,
    TilsScoreWriter,
    open_multiresolutionimage_image,
)

from segmentationpredictor import seg_predict


def process_image_tile_to_segmentation(
    image_slide,
    mask_slide,
    writer: SegmentationWriter,
    dimensions: Tuple[int, int],

    tile_size: int = 128,
    tile_step: int = 256,
    kernel_size: int = 512,
    channel: int = 8,
    batch_size: int = 4,
    downsample: int = 16,
) -> Any:
    MODEL_PATH = './segmentationpredictor/v5-2-17-jit.pth'
    if torch.cuda.is_available():
        DEVICE = 'cuda:0' 
        print('Using ' + torch.cuda.get_device_name(0) + '\n')
    else:
        DEVICE = 'cpu'
    assert os.path.exists(MODEL_PATH)
    model = torch.jit.load(MODEL_PATH).eval().to(DEVICE)
    # 逐行预测与融合
    with torch.no_grad():
        results = seg_predict(
            model=model,  # 模型文件
            image=image_slide,  # 输入流
            mask=mask_slide,  # 辅助输入流
            writer=writer,  # 输出流
            dimensions=dimensions,  # 维度数
            tile_size=tile_size,  # 写图核尺寸
            tile_step=tile_step,  # 融合步长(tile_size 的整数倍)
            ksize=kernel_size,  # 预测核尺寸
            channel=channel,  # 输出通道数
            batch_size=batch_size,
            device=DEVICE,  # 预测所用容器
            downsample=downsample,  # 返回结果的降采样倍数
        )
    return results


def process():
    """Proceses a test slide"""
    from utils.opencv_utils import OpenCV

    level = READING_LEVEL
    tile_size = WRITING_TILE_SIZE  # should be a power of 2

    initialize_output_folders()

    # get input paths
    image_path = get_image_path_from_input_folder()
    tissue_mask_path = get_tissue_mask_path_from_input_folder()

    file_basename,_ = os.path.splitext(os.path.basename(image_path))
    print(f'Processing image: {image_path}')
    print(f'Processing with mask: {tissue_mask_path}')

    # open images
    image = open_multiresolutionimage_image(path=image_path)
    tissue_mask = open_multiresolutionimage_image(path=tissue_mask_path)
    # get image info
    dimensions = image.getDimensions()
    spacing = image.getSpacing()

    # create writers
    segmentation_writer = SegmentationWriter(
        TMP_SEGMENTATION_OUTPUT_PATH,
        tile_size=tile_size,
        dimensions=dimensions,
        spacing=spacing,
    )
    detection_writer = DetectionWriter(TMP_DETECTION_OUTPUT_PATH)
    tils_score_writer = TilsScoreWriter(TMP_TILS_SCORE_PATH)

    print("Processing image...")
    
    process_level = 2
    level_downsamples = round(image.getLevelDownsample(process_level))
    
    process_image_tile_to_segmentation(
        image_slide=image,
        mask_slide=tissue_mask,
        writer=segmentation_writer,
        dimensions=dimensions,
        tile_size=tile_size,
        tile_step= 256,
        kernel_size= 512,
        channel= 8,
        batch_size= 4,
        downsample= level_downsamples,
    )

    # # if TMP_SEGMENTATION_OUTPUT_PATH.suffix != '.tif':
    # #     TMP_SEGMENTATION_OUTPUT_PATH = TMP_SEGMENTATION_OUTPUT_PATH / '.tif'

    # # save segmentation
    segmentation_writer.save()
    
    fast_mode = True
    tissue_mask_thumb = tissue_mask.getUCharPatch(
                startX=0, startY=0, width = image.getLevelDimensions(process_level)[0], 
                height = image.getLevelDimensions(process_level)[1], level=process_level).squeeze()
    tissue_mask_area = np.sum(tissue_mask_thumb>0)*(level_downsamples/1000)**2*spacing[0]*spacing[1]
    if tissue_mask_area < 5:
       fast_mode = False
    del tissue_mask,tissue_mask_thumb
    seg_mask = open_multiresolutionimage_image(path=str(TMP_SEGMENTATION_OUTPUT_PATH))
    
    results_ori = seg_mask.getUCharPatch(
                startX=0, startY=0, width = image.getLevelDimensions(process_level)[0], 
                height = image.getLevelDimensions(process_level)[1], level=process_level).squeeze()
    print("\n" + " the shape of segmentation result array is ")
    print(results_ori.shape)
    
    results = get_tumor_bulk(results_ori,
                        dimensions, level_downsamples,
                        contour_area_threshold = 0.05,
                        invasive_tumor_threshold_list = [0.05,0.95])
    
    if fast_mode:
        results_before = results * ((results_ori==2) | (results_ori==6))
    else:
        results_before = results_ori
    
    # np.save(f"./{file_basename}_segmentation_result.npy",results)
    ####################
   
    # if np.sum(results.shape) != np.sum(image.getLevelDimensions(process_level)[1],
    #                                    image.getLevelDimensions(process_level)[0]):
    #     results = results[:min(image.getLevelDimensions(process_level)[1],results.shape[0]),
    #                     :min(image.getLevelDimensions(process_level)[0],results.shape[1])]

    # results_baseline = get_cancer_region_baseline(cancer_only_matrix=(results==1) | (results==2) | (results==6), mask=results>0,triangle_circle=75, filter_rate=0.1)

    whole_result_list = []
    json_file_len = 0
    patch_size = int(NUCLEI_DET_BIG_TILE_SIZE - NUCLEI_DET_MODEL_INPUT_SIZE / 4)
    for y in tqdm(range(0, dimensions[1], patch_size),desc='detection predicting and fast mode is ' + str(fast_mode)):
        for x in range(0, dimensions[0], patch_size):
            image_tile = image.getUCharPatch(startX=x,#max(x - NUCLEI_DET_MODEL_INPUT_SIZE, 0),
                                             startY=y,#max(y - NUCLEI_DET_MODEL_INPUT_SIZE, 0),
                                             width=NUCLEI_DET_BIG_TILE_SIZE,
                                             height=NUCLEI_DET_BIG_TILE_SIZE, level=level)
            tissue_mask_tile = results_before[
                               round(y / level_downsamples):min(round((y + NUCLEI_DET_BIG_TILE_SIZE) / level_downsamples), image.getLevelDimensions(process_level)[1]),
                               round(x / level_downsamples):min(round((x + NUCLEI_DET_BIG_TILE_SIZE) / level_downsamples), image.getLevelDimensions(process_level)[0])]
            if np.sum(tissue_mask_tile) > 0:
                if fast_mode:
                    tissue_mask_tile = OpenCV(tissue_mask_tile).resize(image_tile.shape[1], image_tile.shape[0])
                    detections = loader_prediction(image_tile,tissue_mask_tile,NUCLEI_DET_MODEL_INPUT_SIZE,NUCLEI_DET_TILE_STEP,0.3)
                else:
                    tissue_mask_tile = np.uint8((tissue_mask_tile==2) | (tissue_mask_tile==6))
                    tissue_mask_tile = OpenCV(tissue_mask_tile).resize(image_tile.shape[1], image_tile.shape[0])
                    detections = loader_prediction(image_tile,np.ones(tissue_mask_tile.shape,dtype=np.uint8),NUCLEI_DET_MODEL_INPUT_SIZE,NUCLEI_DET_TILE_STEP,0.15)
            else:
                continue

            if detections.shape[0] > 0:
                #
                # has_1 = False
                # for p in detections:
                #     # pos = np.int32(np.round(p[:2])).tolist()
                #     prob = float(p[2])
                #     if prob < 0.3:
                #         continue
                    # has_1 = True
                    # image_tile = cv2.circle(image_tile, tuple(pos), 6, (0, 255, 0), 2)
                # if has_1:
                #     image_tile = cv2.resize(image_tile, (768, 768), interpolation=cv2.INTER_NEAREST)
                #     cv2.imshow('tt', image_tile)
                #     cv2.waitKey(1)
                ###
                detections = detections[
                    (detections[:, 0] > NUCLEI_DET_MODEL_INPUT_SIZE /8) & \
                    (detections[:, 1] > NUCLEI_DET_MODEL_INPUT_SIZE /8) & \
                    (detections[:, 0] <= NUCLEI_DET_BIG_TILE_SIZE - NUCLEI_DET_MODEL_INPUT_SIZE /8) \
                    & (detections[:, 1] <= NUCLEI_DET_BIG_TILE_SIZE - NUCLEI_DET_MODEL_INPUT_SIZE /8)]
                final_list = []
                for f in range(detections.shape[0]):
                    top = max(0,int(detections[f,1] -2 * level_downsamples))
                    bottom = min(tissue_mask_tile.shape[0], int(detections[f, 1] + 2 * level_downsamples))
                    left = max(0,int(detections[f,0] - 2 * level_downsamples))
                    right = min(tissue_mask_tile.shape[1], int(detections[f, 0] + 2 * level_downsamples))
                    # if np.sum((tissue_mask_tile[top:bottom,left:right] == 2) | (tissue_mask_tile[top:bottom,left:right] == 6)) > 0 :
                    if np.sum(tissue_mask_tile[top:bottom,left:right]) > 0:
                        final_list.append(detections[f,:].tolist())
                        json_file_len += 1
                if len(final_list) > 0:
                    detection_writer.write_detections(
                            detections=final_list, spacing=spacing, x_offset=x,
                            y_offset=y
                        )
                # json_file_len += len(detections)
                # tmp_array = final_detections.copy()
                detections[:, 0] += x#x - NUCLEI_DET_MODEL_INPUT_SIZE if x > NUCLEI_DET_MODEL_INPUT_SIZE else x

                detections[:, 1] += y#y - NUCLEI_DET_MODEL_INPUT_SIZE if y > NUCLEI_DET_MODEL_INPUT_SIZE else y
                whole_result_list.append(detections)
            del detections
    
    if len(whole_result_list) > 0:
        print('the whole lenth of json file:' + str(json_file_len))
        try:
            nuclei_det_prediction = np.vstack([x for x in whole_result_list])
            print("\n" + " the shape of nuclei_info array is ")
            print(nuclei_det_prediction.shape)
            # np.save(f"./{file_basename}_nuclei_det_prediction_0.3.npy",nuclei_det_prediction)
        except Exception as e:
            print(e)
    # save detection
    detection_writer.save()
    # del whole_result_list

    ####################
    # results = results_baseline * ((results==2) | (results==6))
    # del results_baseline
    if not fast_mode:
        results_before = results * ((results_ori==2) | (results_ori==6))

    # results_after = get_tumor_bulk_new(results_ori,results,level_downsamples,spacing,kernel_size=(100, 100))
    # from utils.tils_score_calc import tumor_bulk_save
    # from pathlib import Path
    # tumor_bulk_writer = SegmentationWriter(
    #     Path(f"./{file_basename}_200.tif"),
    #     tile_size=tile_size,
    #     dimensions=dimensions,
    #     spacing=spacing,
    # )
    # tumor_bulk_save(results,tumor_bulk_writer,dimensions,level_downsamples,256)
    # tumor_bulk_writer.save()
    
    print("Compute tils score...")
    # compute tils score
    # tils_score = process_segmentation_detection_to_tils_score(
    #     TMP_SEGMENTATION_OUTPUT_PATH, detection_writer.detections
    # )
    if len(whole_result_list) > 0:
        try:
            
            tils_score = get_score(results_before,nuclei_det_prediction,
                                dimensions,level_downsamples,256)
            tils_score = round(max(0.01,tils_score/spacing[0]/spacing[1])*100,3)
        except Exception as e:
            print(e)
            tils_score = 0
    else:
        tils_score = 0
    tils_score_writer.set_tils_score(tils_score=min(100,tils_score))
    print('tils score is ' + str(tils_score))
    print("Saving...")
    # save tils score
    tils_score_writer.save()

    print("Copy data...")
    # save data to output folder
    copy_data_to_output_folders()

    print("Done!")

