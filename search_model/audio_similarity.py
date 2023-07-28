import numpy as np
from numba import njit
import numba.np.unsafe.ndarray

@njit
def get_cs(array1, array2):

    dot_XY = np.dot(array1.reshape(1, -1), array2.reshape(-1, 1))[0][0]
    norm_X = np.sqrt(np.sum(array1**2))
    norm_Y = np.sqrt(np.sum(array2**2))
    cs = dot_XY / (norm_X * norm_Y)

    return cs

@njit
def get_all_cs(array1, array2):
    cs_array = np.zeros((array2.shape[0], 1))
    for i in range(array2.shape[0]):
        cs_array[i] = get_cs(array1, array2[i, :])
    return cs_array