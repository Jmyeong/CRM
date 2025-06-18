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
import torch.nn.functional as F
cv2.setNumThreads(0)

def inpaint_color(color, mask, N=5):
    # 랜덤 색상 생성 (B, G, R)
    inpainted_color_list = []
    mask = np.array(mask)
    
    fill_color = np.random.randint(0, 256, 3, dtype=np.uint8)
    
    # 원본 이미지 복사 후 마스크 영역에 랜덤 색 적용
    inpainted_color = np.array(color.copy())
    
    inpainted_color[mask > 0] = fill_color
    # inpainted_color_list.append(Image.fromarray(inpainted_color))
    inpainted_color = Image.fromarray(inpainted_color)
    return inpainted_color

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
                 use_depth_hints,
                 use_glass_hints,
                 depth_hint_path=None,
                 glass_hint_path=None,
                 is_train=False,
                 img_ext='.png'):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.use_depth_hints = use_depth_hints
        self.use_glass_hints = True
        # assume depth hints npys are stored in data_path/depth_hints unless specified
        if depth_hint_path is None:
            self.depth_hint_path = os.path.join(self.data_path, 'depth_hints')
        else:
            self.depth_hint_path = depth_hint_path
            
        self.interp = Image.LANCZOS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        self.train_student = True
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

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
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
        inputs["path"] = self.filenames[index]
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

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()
            # print(K)
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
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.train_student:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            depth_gt_for_valid = self.get_depth_for_valid(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))
            inputs["depth_gt_for_valid"] = np.expand_dims(depth_gt_for_valid, 0)
            inputs["depth_gt_for_valid"] = torch.from_numpy(inputs["depth_gt_for_valid"].astype(np.float32))
        
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
            
            if self.use_glass_hints:                
                ### Load mask
                side_folder = 'image_02' if side == 'l' else 'image_03'

                # print(folder)
                mask_folder = os.path.join("/ssd1/jm_data/Grounded-Segment-Anything/outputs", folder, side_folder, "mask"
                                           , str(frame_index).zfill(10) + '.png')
                # print(mask_folder)
                # try:
                if os.path.exists(mask_folder):
                    mask = cv2.imread(mask_folder, cv2.IMREAD_GRAYSCALE)
                    mask = cv2.resize(mask, (640, 360))
                # except FileNotFoundError:
                #     raise FileNotFoundError("Warning - cannot find mask for {} {} {}! "
                #                             "Either specify the correct path in option "
                #                             "--mask_path, or run precompute_depth_hints.py to"
                #                             "train with depth hints".format(folder, side_folder,
                #                                                             frame_index))
                else:
                    mask = np.zeros((360, 640))
                if do_flip:
                    mask = np.fliplr(mask)
                mask = np.where(mask > 50, 255, 0)
                mask = torch.from_numpy(mask.copy()).float().unsqueeze(0)
                inputs['mask'] = (mask > 0).float()
                # print(torch.unique(inputs["mask"]))
            else:
                inputs['mask'] = 0
        else:
            ### Load mask
            side_folder = 'image_02' if side == 'l' else 'image_03'

            # print(folder)
            mask_folder = os.path.join("/ssd1/jm_data/Grounded-Segment-Anything/outputs", folder, side_folder, "mask"
                                        , str(frame_index).zfill(10) + '.png')
            # print(mask_folder)
            # try:
            if os.path.exists(mask_folder):
                mask = cv2.imread(mask_folder, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (640, 360))
            # except FileNotFoundError:
            #     raise FileNotFoundError("Warning - cannot find mask for {} {} {}! "
            #                             "Either specify the correct path in option "
            #                             "--mask_path, or run precompute_depth_hints.py to"
            #                             "train with depth hints".format(folder, side_folder,
            #                                                             frame_index))
            else:
                mask = np.zeros((360, 640))
            if do_flip:
                mask = np.fliplr(mask)
            mask = np.where(mask > 50, 255, 0)
            mask = torch.from_numpy(mask.copy()).float().unsqueeze(0)
            inputs['mask'] = (mask > 0).float()
            # print(torch.unique(inputs["mask"]))
                



        return inputs

    def detect_vertical_gaps(self, disparity_map, threshold=1.0, min_height=0):
        """
        disparity 맵에서 유리문과 같이 세로로 긴 영역을 검출하는 함수.
        disparity_map: (H, W) numpy array, disparity 맵
        threshold: disparity 값 차이가 이 값 이상인 경우, 해당 영역을 중요한 영역으로 간주
        min_height: 검출된 영역의 최소 세로 길이
        
        return: 세로로 긴 영역 마스크
        """
        # 1. 큰 차이를 보이는 영역 찾기 (disparity 차이가 threshold 이상)
        diff_map = np.abs(np.diff(disparity_map, axis=1))  # 가로 방향으로 차이 계산
        diff_map = np.pad(diff_map, ((0, 0), (1, 0)), mode='constant', constant_values=0)  # 원래 크기 맞추기
        
        # threshold 이상의 차이를 갖는 영역을 마스크로 추출
        mask = diff_map > threshold
        
        # 2. 세로로 긴 영역 검출
        # mask에서 세로로 긴 영역을 찾아냄 (세로 길이가 min_height 이상)
        vertical_mask = np.zeros_like(mask, dtype=np.uint8)
        
        for col in range(mask.shape[1]):
            start_row = None
            for row in range(mask.shape[0]):
                if mask[row, col] == 1:
                    if start_row is None:
                        start_row = row
                else:
                    if start_row is not None and row - start_row >= min_height:
                        vertical_mask[start_row:row, col] = 1
                    start_row = None
                    
            # 마지막에 끝나는 부분도 처리
            if start_row is not None and mask.shape[0] - start_row >= min_height:
                vertical_mask[start_row:, col] = 1

        return vertical_mask


    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
    
    def get_depth_for_valid(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
    
    def get_mask(self, folder ,frame_index, side, do_flip):
        raise NotImplementedError
