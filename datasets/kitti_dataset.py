# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

# 2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000000.png

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset


class KITTIDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.
        # self.K = np.array([[0.58, 0, 0.5, 0],
        #                    [0, 1.92, 0.5, 0],
        #                    [0, 0, 1, 0],
        #                    [0, 0, 0, 1]], dtype=np.float32)

        # zed2i
        # w, h = 1242, 375
        w,h = 640, 360
        self.K = np.array([[260.8747863769531 / w, 0, 321.9953308105469 / w, 0],
                           [0, 260.8747863769531 / h, 179.68511962890625 / h, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        
        self.full_res_shape = (w, h)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
        self.train_mode = "student"
    def check_depth(self):
        # print(self.filenames[0])
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "ouster_bins/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        # print(frame_index, side)
        # print(self.get_image_path(folder, frame_index, side))
        # print(folder)
        # print(self.get_image_path(folder, frame_index, side).split("/")[-1].split("_")[-1])
        
        # if self.get_image_path(folder, frame_index, side).split("/")[9] == "mask":
            
        if self.get_image_path(folder, frame_index, side).split("/")[7] == "inpaints":
            glass_path = self.get_image_path(folder, frame_index, side)
            image_path = glass_path.replace(f"{int(frame_index)}.png", f"{int(frame_index)}_label.png")
            # print(image_path)

            if os.path.exists(image_path):
                color = self.loader(image_path)
                print(color.width, color.height)
            else:
                color = np.zeros((360, 640, 3), dtype=np.uint8)
                # print(color.shape)
                color = pil.fromarray(color)
                # print(color.width)
        else: 
            # print(folder, frame_index, side)
            # print(self.get_image_path(folder, frame_index, side))
            # if os.path.exists(self.get_image_path(folder, frame_index, side)):
            color = self.loader(self.get_image_path(folder, frame_index, side))
            # else:
            #     color = np.zeros((360, 640, 3), dtype=np.uint8)
            #     color = pil.fromarray(color)
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_mask(self, folder, frame_index, side, do_flip):
        if self.get_mask_path(folder, frame_index, side).split("/")[9] == "mask":
            glass_mask_path = self.get_mask_path(folder, frame_index, side)
            glass_mask_path = glass_mask_path.replace(f"{int(frame_index)}.png", f"{int(frame_index)}_label.png")
            if os.path.exists(glass_mask_path):
                # print(glass_mask_path)
                mask = self.loader(glass_mask_path)
                # print(np.unique(np.array(mask)))
            else:
                mask = np.zeros((360, 640), dtype=np.uint8)
                mask = pil.fromarray(mask)
        else:
            mask = self.loader(self.get_mask_path(folder, frame_index, side))
    
        if do_flip:
            mask = mask.transpose(pil.FLIP_LEFT_RIGHT)
        
        mask = mask.convert("L")

        return mask
    
class KITTIRAWDataset(KITTIDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(KITTIRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        # print(image_path)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))
        print(f"velo_filename : {velo_filename}")
        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt


class KITTIOdomDataset(KITTIDataset):
    """KITTI dataset for odometry training and testing
    """
    def __init__(self, *args, **kwargs):
        super(KITTIOdomDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            "sequences/{:02d}".format(int(folder)),
            "image_{}".format(self.side_map[side]),
            f_str)
        return image_path


class KITTIDepthDataset(KITTIDataset):
    """KITTI dataset which uses the updated ground truth depth maps
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDepthDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            folder,
            "image_0{}/data".format(self.side_map[side]),
            f_str)
        return image_path
    
    def get_mask_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        # print(folder)
        mask_path = os.path.join(
            self.data_path,
            folder,
            "mask/image_0{}/data".format(self.side_map[side]),
            f_str)
        return mask_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:010d}.png".format(frame_index)
        depth_path = os.path.join(
            self.data_path,
            folder,
            "proj_depth/groundtruth/image_0{}".format(self.side_map[side]),
            f_str)
        # print(depth_path)
        depth_gt = pil.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) 
        # print(depth_gt.shape)
        # print(f"gt : {np.unique(depth_gt)}")
        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

class JBNUDepthDataset(KITTIDataset):
    """KITTI dataset which uses the updated ground truth depth maps
    """
    def __init__(self, *args, **kwargs):
        super(JBNUDepthDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            folder,
            "image_0{}/data".format(self.side_map[side]),
            f_str)
        return image_path
    
    def get_mask_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext).replace(".png", "_label.png")
        # print(folder)
        mask_path = os.path.join(
            self.data_path,
            folder,
            "mask/image_0{}/data".format(self.side_map[side]),
            f_str)
        # print(mask_path)
        return mask_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:010d}.npy".format(frame_index)
        if self.train_mode == "teacher":
            depth_path = os.path.join(
                self.data_path,
                folder,
                "proj_depth/groundtruth/image_0{}".format(self.side_map[side]),
                f_str)
        elif self.train_mode == "student":
            depth_path = os.path.join(
                self.data_path,
                folder,
                "proj_depth/groundtruth_checked/image_0{}".format(self.side_map[side]),
                f_str)

        # print(depth_path)
        depth_gt = np.load(depth_path)
        # depth_gt = pil.open(depth_path)
        depth_gt = pil.fromarray(depth_gt)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) 

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
    
    def get_depth_for_valid(self, folder, frame_index, side, do_flip):
        f_str = "{:010d}.npy".format(frame_index)
        depth_path = os.path.join(
            self.data_path,
            folder,
            "proj_depth/groundtruth/image_0{}".format(self.side_map[side]),
            f_str)
        # print(depth_path)
        depth_gt = np.load(depth_path)
        # depth_gt = pil.open(depth_path)
        depth_gt = pil.fromarray(depth_gt)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) 

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

