import glob
import inspect
import logging
import os
import shutil

import cv2
import numpy as np
from scipy import ndimage


def normalize(mask, dtype=np.uint8):
    return (255 * mask / np.amax(mask)).astype(dtype)  # amax返回数组中的最大值


def get_bounding_box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]


def cropping_center(x, crop_shape, batch=False):
    orig_shape = x.shape
    if not batch:
        h0 = int((orig_shape[0] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[1] - crop_shape[1]) * 0.5)
        x = x[h0:h0 + crop_shape[0], w0:w0 + crop_shape[1]]
    else:
        h0 = int((orig_shape[1] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[2] - crop_shape[1]) * 0.5)
        x = x[:, h0: h0 + crop_shape[0], w0: w0 + crop_shape[1]]
    return x


def rm_n_mkdir(dir_path):
    """Remove and make directory."""
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

