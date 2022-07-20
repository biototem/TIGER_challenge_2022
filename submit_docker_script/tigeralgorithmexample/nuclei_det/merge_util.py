import cv2
import numpy as np
from rw import NUCLEI_DET_MODEL_INPUT_SIZE

kernel = cv2.getGaussianKernel(NUCLEI_DET_MODEL_INPUT_SIZE, round(NUCLEI_DET_MODEL_INPUT_SIZE/4))
kernel /= np.average(kernel)
kernel = np.matmul(kernel,kernel.T)
# kernel = np.expand_dims(kernel, axis=2)

def merger_method(pred_det_pm,pred_det_2_pm,
                  whole_pred_det_pm_mask,whole_pred_det_2_pm_mask,
                  x_coord,y_coord,
                  method = "Average"):
    '''
    :param pred_det_pm:                         粗热图，(64,64,1)
    :param pred_det_2_pm:                       细热图,(64,64,1)
    :param whole_pred_det_pm_mask:              原预测大图大小通道数为2的矩阵；第0通道存放粗热图叠加融合后的数值；第1通道存对应位置放像素点被叠加融合的次数
    :param whole_pred_det_2_pm_mask:            原预测大图大小通道数为2的矩阵；第0通道存放细热图叠加融合后的数值；第1通道存对应位置放像素点被叠加融合的次数
    :param x_coord:                             当前热图在预测大图中的左上角横坐标
    :param y_coord:                             当前热图在预测大图中的左上角纵坐标
    :param method:                              融合方法标识，只能是"Gaussian"或"Average"
    :return:                                    处理后的whole_pred_det_pm_mask和whole_pred_det_2_pm_mask
    '''
    if method not in ["Gaussian","Average"]:
        method = "Average"
    if method == "Average":
        whole_pred_det_pm_mask[y_coord:y_coord + NUCLEI_DET_MODEL_INPUT_SIZE,
                                x_coord:x_coord + NUCLEI_DET_MODEL_INPUT_SIZE,0] += pred_det_pm[:, :, 0]
        whole_pred_det_pm_mask[y_coord:y_coord + NUCLEI_DET_MODEL_INPUT_SIZE,
                                x_coord:x_coord + NUCLEI_DET_MODEL_INPUT_SIZE, 1] += 1
        whole_pred_det_2_pm_mask[y_coord:y_coord + NUCLEI_DET_MODEL_INPUT_SIZE,
                                x_coord:x_coord + NUCLEI_DET_MODEL_INPUT_SIZE, 0] += pred_det_2_pm[:, :, 0]
        whole_pred_det_2_pm_mask[y_coord:y_coord + NUCLEI_DET_MODEL_INPUT_SIZE,
                                x_coord:x_coord + NUCLEI_DET_MODEL_INPUT_SIZE, 1] += 1
    elif method == "Gaussian":
        whole_pred_det_pm_mask[y_coord:y_coord + NUCLEI_DET_MODEL_INPUT_SIZE,
                                x_coord:x_coord + NUCLEI_DET_MODEL_INPUT_SIZE, 0] += pred_det_pm[:, :, 0] * kernel
        whole_pred_det_pm_mask[y_coord:y_coord + NUCLEI_DET_MODEL_INPUT_SIZE,
                                x_coord:x_coord + NUCLEI_DET_MODEL_INPUT_SIZE, 1] += kernel
        whole_pred_det_2_pm_mask[y_coord:y_coord + NUCLEI_DET_MODEL_INPUT_SIZE,
                                x_coord:x_coord + NUCLEI_DET_MODEL_INPUT_SIZE, 0] += pred_det_2_pm[:, :, 0] * kernel
        whole_pred_det_2_pm_mask[y_coord:y_coord + NUCLEI_DET_MODEL_INPUT_SIZE,
                                x_coord:x_coord + NUCLEI_DET_MODEL_INPUT_SIZE, 1] += kernel
    return whole_pred_det_pm_mask,whole_pred_det_2_pm_mask
