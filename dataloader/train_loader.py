import csv
import glob
import os
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch.utils.data

import imgaug as ia
from imgaug import augmenters as iaa
from misc.utils import cropping_center

from .augs import (
    add_to_brightness,
    add_to_contrast,
    add_to_hue,
    add_to_saturation,
    gaussian_blur,
    median_blur,
)


####
class FileLoader(torch.utils.data.Dataset):
    """Data Loader. Loads images from a file list and 
    performs augmentation with the albumentation library.
    After augmentation, horizontal and vertical maps are 
    generated.

    Args:
        file_list: list of filenames to load
        input_shape: shape of the input [h,w] - defined in config.py
        mask_shape: shape of the output [h,w] - defined in config.py
        mode: 'train' or 'valid'
        
    """

    # TODO: doc string

    def __init__(
        self,
        file_list,  # 那一堆npy文件
        with_type=False,  # False
        input_shape=None,  # [270, 270]
        mask_shape=None,  # [80, 80]
        mode="train",  # 'train'
        setup_augmentor=True,  # True
        target_gen=None,  # gen_target {}
    ):
        assert input_shape is not None and mask_shape is not None
        self.mode = mode
        self.info_list = file_list  # .npy
        self.with_type = with_type  # False
        self.mask_shape = mask_shape  # [80, ]
        self.input_shape = input_shape  # [270, ]
        self.id = 0
        self.target_gen_func = target_gen[0]  # gen_target 在target.py里面
        self.target_gen_kwargs = target_gen[1]  # {}
        if setup_augmentor:  # True
            self.setup_augmentor(0, 0)
        return

    def setup_augmentor(self, worker_id, seed):  # 0， 0
        self.augmentor = self.__get_augmentation(self.mode, seed)
        self.shape_augs = iaa.Sequential(self.augmentor[0])
        self.input_augs = iaa.Sequential(self.augmentor[1])
        self.id = self.id + worker_id  # 0+0=0
        return

    def __len__(self):
        return len(self.info_list)

    def __getitem__(self, idx):
        path = self.info_list[idx]
        data = np.load(path)

        # split stacked channel into image and label
        img = (data[..., :3]).astype("uint8")  # RGB images
        ann = (data[..., 3:]).astype("int32")  # instance ID map and type map

        if self.shape_augs is not None:
            shape_augs = self.shape_augs.to_deterministic()
            img = shape_augs.augment_image(img)
            ann = shape_augs.augment_image(ann)

        if self.input_augs is not None:
            input_augs = self.input_augs.to_deterministic()
            img = input_augs.augment_image(img)

        img = cropping_center(img, self.input_shape)
        feed_dict = {"img": img}

        inst_map = ann[..., 0]  # HW1 -> HW
        if self.with_type:
            type_map = (ann[..., 1]).copy()
            type_map = cropping_center(type_map, self.mask_shape)
            #type_map[type_map == 5] = 1  # merge neoplastic and non-neoplastic
            feed_dict["tp_map"] = type_map

        # TODO: document hard coded assumption about #input
        target_dict = self.target_gen_func(
            inst_map, self.mask_shape, **self.target_gen_kwargs
        )
        feed_dict.update(target_dict)

        return feed_dict

    # 食用imgaug库来进行图像增强augmentation
    def __get_augmentation(self, mode, rng):  # train 0
        if mode == "train":
            # shape augmentation
            shape_augs = [
                # * order = ``0`` -> ``cv2.INTER_NEAREST``
                # * order = ``1`` -> ``cv2.INTER_LINEAR``
                # * order = ``2`` -> ``cv2.INTER_CUBIC``
                # * order = ``3`` -> ``cv2.INTER_CUBIC``
                # * order = ``4`` -> ``cv2.INTER_CUBIC``
                # ! for pannuke v0, no rotation or translation, just flip to avoid mirror padding
                iaa.Affine(  # 包括缩放，平移，剪切和旋转
                    # scale控制图像在x、y轴上的缩放比例， 范围是0.8 - 1.2
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    # translate_percent 控制图像在x、y上的平移比例 范围是-0.01 - 0.01
                    translate_percent={"x": (-0.01, 0.01), "y": (-0.01, 0.01)},
                    # shear设置剪切角度范围 -5° - 5°
                    shear=(-5, 5),  # shear by -5 to +5 degrees
                    # rotate设置旋转角度范围 -179 - +179 degree
                    rotate=(-179, 179),  # rotate by -179 to +179 degrees
                    # order=0 使用最临近插值的方法
                    order=0,  # use nearest neighbour
                    # backend = 'cv2' 使用opencv作为处理后端
                    backend="cv2",  # opencv for fast processing
                    seed=rng,  # seed = rng = 0
                ),
                # set position to 'center' for center crop
                # else 'uniform' for random crop
                # 裁剪为特定的尺寸
                iaa.CropToFixedSize(
                    self.input_shape[0], self.input_shape[1], position="center"
                ),  # 270 270 center
                iaa.Fliplr(0.5, seed=rng),  # 以0.5的概率左右翻转
                iaa.Flipud(0.5, seed=rng),  # 以0.5的概率上下翻转
            ]
            # input augmentation???
            input_augs = [
                iaa.OneOf(  # 从提供的选择一个
                    [
                        iaa.Lambda(  # 高斯模糊
                            seed=rng,
                            func_images=lambda *args: gaussian_blur(*args, max_ksize=3),
                        ),
                        iaa.Lambda(  # 中值模糊
                            seed=rng,
                            func_images=lambda *args: median_blur(*args, max_ksize=3),
                        ),
                        iaa.AdditiveGaussianNoise(  # 加性高斯噪声
                            loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                        ),
                    ]
                ),
                iaa.Sequential(
                    [
                        iaa.Lambda(  # 调整色调
                            seed=rng,
                            func_images=lambda *args: add_to_hue(*args, range=(-8, 8)),
                        ),
                        iaa.Lambda(  # 调整饱和度
                            seed=rng,
                            func_images=lambda *args: add_to_saturation(
                                *args, range=(-0.2, 0.2)
                            ),
                        ),
                        iaa.Lambda(  # 调整亮度
                            seed=rng,
                            func_images=lambda *args: add_to_brightness(
                                *args, range=(-26, 26)
                            ),
                        ),
                        iaa.Lambda(  # 调整对比度
                            seed=rng,
                            func_images=lambda *args: add_to_contrast(
                                *args, range=(0.75, 1.25)
                            ),
                        ),
                    ],
                    random_order=True,
                ),
            ]
        elif mode == "valid":
            shape_augs = [
                # set position to 'center' for center crop
                # else 'uniform' for random crop
                iaa.CropToFixedSize(
                    self.input_shape[0], self.input_shape[1], position="center"
                )
            ]
            input_augs = []

        return shape_augs, input_augs
