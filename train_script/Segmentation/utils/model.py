import numpy as np
import cv2
import torch


def one_hot_numpy(label, num):
    return np.eye(num, dtype=np.uint8)[label]


def one_hot_torch(label, num):
    return torch.eye(num, dtype=torch.uint8)[label]


def argmax_numpy(predict, axis):
    return one_hot_numpy(np.argmax(predict, axis=axis), num=predict.shape[axis])


def gaussian_kernel(size=3, steep=2):
    """
    provide an square matrix that matches the gaussian function
    this may used like an kernel of weight
    """
    x = cv2.getGaussianKernel(ksize=size, sigma=size / steep)
    # the numbers are too small ~ and there is no influence on multiple
    x /= np.average(x)
    return np.matmul(x, x.T)
