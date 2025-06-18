# Copyright Niantic 2020. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the DepthHints licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed
import cv2

import torch
import torch.utils.data as data
from torchvision import transforms

from glob import glob

cv2.setNumThreads(0)


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        use_depth_hints
        use_glass_hints
        depth_hint_path
        glass_hint_path
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 use_depth_hints=None,
                 use_glass_hints=None,
                 depth_hint_path=None,
                 glass_hint_path=None,
                 is_train=False,
                 img_ext='.png'):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        # print(f"filenames : {filenames}")
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.use_depth_hints = use_depth_hints
        self.use_glass_hints = use_glass_hints
        # print(f"use glass hints : {use_glass_hints}")
        # assume depth hints npys are stored in data_path/depth_hints unless specified
        if depth_hint_path is None:
            self.depth_hint_path = os.path.join(self.data_path, 'depth_hints')
        else:
            self.depth_hint_path = depth_hint_path

        if glass_hint_path is None:
            self.glass_hint_path = os.path.join(self.data_path, 'glass_hints')
        else:
            self.glass_hint_path = glass_hint_path
            
        self.interp = Image.LANCZOS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
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

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])
            if "color_glass" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])
        for k in list(inputs):
            f = inputs[k]
            # print(k)
            if "color" in k:
                n, im, i = k
                # print(k)
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))
            if "color_glass" in k:
                n, im, i = k
                # print(k)
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps
            "depth_hint"                            for depth hint
            "depth_hint_mask"                       for mask of valid depth hints

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()

        folder = line[0]
        glass_folder = "inpaints/" + line[0]
        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0
        # print(frame_g_index, frame_index)
        if len(line) == 3:
            side = line[2]
        else:
            side = None

        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
                if self.use_glass_hints:
                    inputs[("color_glass", i, -1)] = self.get_color(glass_folder, frame_index, other_side, do_flip)
                    # print(inputs[("color_glass", i, -1)])
                    inputs['glass_mask'] = self.get_mask(folder, frame_index, other_side, do_flip)
                else:
                    inputs[("color_glass", i, -1)] = Image.fromarray(np.zeros((360, 640), dtype=np.float32))
                    inputs['glass_mask'] = Image.fromarray(np.zeros((360, 640), dtype=np.uint8))
            else:
                inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)
                if self.use_glass_hints:
                    inputs[("color_glass", i, -1)] = self.get_color(glass_folder, frame_index + i, side, do_flip)
                    inputs['glass_mask'] = self.get_mask(folder, frame_index + i, side, do_flip)
                else:
                    inputs[("color_glass", i, -1)] = Image.fromarray(np.zeros((360, 640), dtype=np.float32))
                    inputs['glass_mask'] = Image.fromarray(np.zeros((360, 640), dtype=np.uint8))
            # print(np.unique(np.array(inputs['glass_mask'])))
            # print(np.max(inputs["glass_mask"]))
            if np.max(inputs["glass_mask"]) == 255:
                inputs["mask_index"] = 1
                # inputs[("color", i, -1)] = inputs[("color_glass", i, -1)]
            else:
                inputs["mask_index"] = 0
            
            inputs['glass_mask'] = self.to_tensor(inputs['glass_mask'])
            # print(torch.unique(inputs['glass_mask']))
        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            # print(i)
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]
            # if self.use_glass_hints:
            #     del inputs[("color_glass", i, -1)]

                
        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.12

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

            # load depth hint
            if self.use_depth_hints:
                side_folder = 'image_02' if side == 'l' else 'image_03'
                depth_folder = os.path.join(self.depth_hint_path, folder, side_folder,
                                            str(frame_index).zfill(10) + '.npy')

                try:
                    depth = np.load(depth_folder)[0]
                except FileNotFoundError:
                    raise FileNotFoundError("Warning - cannot find depth hint for {} {} {}! "
                                            "Either specify the correct path in option "
                                            "--depth_hint_path, or run precompute_depth_hints.py to"
                                            "train with depth hints".format(folder, side_folder,
                                                                            frame_index))

                if do_flip:
                    depth = np.fliplr(depth)

                depth = cv2.resize(depth, dsize=(self.width, self.height),
                                   interpolation=cv2.INTER_NEAREST)
                inputs['depth_hint'] = torch.from_numpy(depth).float().unsqueeze(0)
                inputs['depth_hint_mask'] = (inputs['depth_hint'] > 0).float()
            
            # Load Glass hints
            if self.use_glass_hints:
                side_folder = 'image_02' if side == 'l' else 'image_03'
                glass_depth_files = glob(self.glass_hint_path+folder+"/"+side_folder + "/*.png", recursive=False)

                depth_g_folder = os.path.join(self.glass_hint_path, folder, side_folder,
                                            str(frame_index).zfill(10) + '.png')
                
                # print(glass_depth_files)
                # print(depth_g_folder)
                depth_g = np.zeros_like(depth)
                try:
                    if depth_g_folder in glass_depth_files:
                        depth_g = cv2.imread(depth_g_folder, cv2.IMREAD_GRAYSCALE)
                        # depth_g = np.load(depth_g_folder)
                        # print(depth_g_folder)
                        # print(np.unique(depth_g))
                        # print(depth_g.shape)

                except FileNotFoundError:
                    raise FileNotFoundError("Warning - cannot find glass hint for {} {} {}! "
                                            "Either specify the correct path in option "
                                            "train with glass hints".format(folder, side_folder,
                                                                            frame_index))
                if do_flip:
                    depth_g = np.fliplr(depth_g)

                depth_g = cv2.resize(depth_g, dsize=(self.width, self.height),
                                interpolation=cv2.INTER_NEAREST)
            
                inputs['glass_hint'] = torch.from_numpy(depth_g).float().unsqueeze(0)
                inputs['glass_hint_mask'] = (inputs['glass_hint'] > 0).float() 
                    
        return inputs

        
        # print(line)
        # glass_mask_folder = 
             
        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
    
    def get_mask(self, folder ,frame_index, side, do_flip):
        raise NotImplementedError
