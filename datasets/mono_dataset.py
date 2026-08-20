
from __future__ import absolute_import, division, print_function

import os
import random

import cv2
import numpy as np
import copy
from PIL import Image

import torch
import torch.utils.data as data
from torchvision import transforms
from datasets.utils import Resize


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class MonoDataset(data.Dataset):
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.jpg'):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.teacher_height = 392
        self.teacher_width = 672
        self.num_scales = num_scales
        self.mean = np.array([[[0.485]], [[0.456]], [[0.406]]]).astype(np.float32)
        self.std = np.array([[[0.229]], [[0.224]], [[0.225]]]).astype(np.float32)
        try:
            self.interp = Image.LANCZOS
        except AttributeError:
            self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.load_depth = self.check_depth()

    def preprocess(self, inputs, color_aug, color_teacher_aug, do_color_aug):
        for k in list(inputs):
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])
                    if i == 0 and im == 0:
                        resized_data = transforms.Resize(
                            (self.teacher_height, self.teacher_width),
                            interpolation=self.interp
                        )(inputs[(n, im, i-1)])
                        inputs[("colorteacher", im, i)] = resized_data

        for k in list(inputs):
            f = inputs[k]
            if "color" in k or "colorteacher" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))
        for k in list(inputs):
            f = inputs[k]
            if "colorteacher" in k:
                n, im, i = k
                if not do_color_aug:
                    inputs[("teacher_contrast", im, i)] = color_teacher_aug(f)
                else:
                    inputs[("teacher_contrast", im, i)] = inputs[("colorteacher", im, i )]


        for k in list(inputs):
            f = inputs[k]
            if "color_aug" in k or "color" in k or "colorteacher" in k or "colorteacher_aug" in k or"teacher_contrast" in k:
                n, im, i = k
                inputs[(n, im, i)] = (f-self.mean)/self.std

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None

        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
            else:
                inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)

        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter(
                brightness=self.brightness,
                contrast=self.contrast,
                saturation=self.saturation,
                hue=self.hue
            )
            color_teacher_aug= (lambda x: x)
        else:
            color_aug = (lambda x: x)
            color_teacher_aug = transforms.ColorJitter(
                brightness=self.brightness,
                contrast=self.contrast,
                saturation=self.saturation,
                hue=self.hue
            )

        self.preprocess(inputs, color_aug, color_teacher_aug, do_color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_depth and not self.is_train:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
