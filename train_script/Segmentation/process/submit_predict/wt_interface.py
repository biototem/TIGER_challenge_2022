import numpy as np


def wt_interface_trans_img(img: np.ndarray) -> object:
    return None


def wt_predict(matrix: np.ndarray) -> np.ndarray:
    return np.zeros(shape=matrix.shape, dtype=bool)
