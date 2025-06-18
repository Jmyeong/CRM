# Copyright Niantic 2020. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the DepthHints licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import time
import random

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from IPython import embed
### +

class MaskSmoothLoss(nn.Module):
    def __init__(self):
        super(MaskSmoothLoss, self).__init__()

    def forward(self, output, mask):
        """
        output: (B, H, W)
        mask:   (B, H, W) - 0 또는 1
        """
        masked_output = output * mask
        mask_sum = mask.sum(dim=(1, 2)) + 1e-7  # Zero-division 방지
        masked_mean = masked_output.sum(dim=(1, 2)) / mask_sum  # (B,)
        
        # 마스크 영역 내 평균값과 출력값 차이의 L2 손실 계산 (스무딩 목적)
        loss = ((masked_output - masked_mean.unsqueeze(1).unsqueeze(2))**2 * mask).sum(dim=(1, 2)) / mask_sum
        return loss.mean()  # 배치 평균

class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target, valid_mask):
        # print("pred : {}".format(pred[("disp", 0)].shape))
        # print("target : {}".format(target[("disp", 0)].shape))
        # print(f"valid_mask : {valid_mask.shape}")
        valid_mask = valid_mask.detach()
        valid_mask = F.interpolate(valid_mask, size=(352, 640), mode="bilinear", align_corners=False)
        valid_mask = valid_mask > 0.5
        losses = []
        for i in range(target[("disp", 0)].shape[0]): 
            # if valid_mask[i].sum() == 0:  # valid_mask가 전부 False이면 스킵
            #     continue 
            # print(target[("disp", 0)][i].shape, valid_mask.shape)
            # print(len(torch.unique(target[("disp", 0)][i][valid_mask[i]])))
            # print(len(torch.unique(pred[("disp", 0)][i][valid_mask[i]])))
            # print()
            diff_log = torch.log(target[("disp", 0)][i]) - torch.log(pred[("disp", 0)][i])
            loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                            self.lambd * torch.pow(diff_log.mean(), 2))
            if len(torch.unique(pred[("disp", 0)][i][valid_mask[i]])) == 0:
                loss = 0.0
            losses.append(loss)
        loss = sum(losses) / len(losses)
        # print(loss)
        return loss

def l1_loss(pred, target, valid_mask=None):
    """
    두 개의 뎁스 맵(pred, target)에 대한 L1 Loss를 계산
    Args:
        pred (torch.Tensor): 예측된 뎁스 맵 (B, 1, H, W)
        target (torch.Tensor): 실제 뎁스 맵 (B, 1, H, W)
        mask (torch.Tensor, optional): 유효한 픽셀을 나타내는 마스크 (B, 1, H, W). 
                                       None이면 모든 픽셀 사용.
    Returns:
        torch.Tensor: L1 Loss 값
    """
    valid_mask = valid_mask.detach()
    valid_mask = F.interpolate(valid_mask, size=(352, 640), mode="bilinear", align_corners=False)
    valid_mask = valid_mask > 0.5
    losses = []
    
    for i in range(target[("disp", 0)].shape[0]):
        loss = torch.abs(pred[("disp", 0)] - target[("disp", 0)])  # L1 Loss 계산
        if valid_mask is not None:
            loss = loss * valid_mask  # 마스크 적용 (유효한 픽셀만 사용)
            losses.append(loss.sum() / (valid_mask.sum() + 1e-6))
        else:
            losses.append(loss.mean)
    loss = sum(losses) / len(losses)
    
    return loss  # 전체 픽셀 평균 Loss

def l2_loss(pred, target, valid_mask=None):
    """
    두 개의 뎁스 맵(pred, target)에 대한 L2 Loss를 계산
    Args:
        pred (torch.Tensor): 예측된 뎁스 맵 (B, 1, H, W)
        target (torch.Tensor): 실제 뎁스 맵 (B, 1, H, W)
        valid_mask (torch.Tensor, optional): 유효한 픽셀을 나타내는 마스크 (B, 1, H, W). 
                                             None이면 모든 픽셀 사용.
    Returns:
        torch.Tensor: L2 Loss 값
    """
    if valid_mask is not None:
        valid_mask = valid_mask.detach()
        valid_mask = F.interpolate(valid_mask, size=(352, 640), mode="bilinear", align_corners=False)
        valid_mask = valid_mask > 0.5  # Thresholding 적용

    losses = []
    
    for i in range(target[("disp", 0)].shape[0]):
        loss = (pred[("disp", 0)] - target[("disp", 0)]) ** 2  # L2 Loss (제곱 오차)

        if valid_mask is not None:
            loss = loss * valid_mask  # 마스크 적용 (유효한 픽셀만 사용)
            losses.append(loss.sum() / (valid_mask.sum() + 1e-6))
        else:
            losses.append(loss.mean())
    loss = sum(losses) / len(losses)

    return loss  # 배치 평균 Loss

def masked_uniformity_loss(disp, valid_mask):
    """
    마스크 내부 깊이 값들이 균일성을 유지하도록 만드는 Loss
    
    Args:
        depth (torch.Tensor): (B, 1, H, W) 깊이 맵
        mask (torch.Tensor): (B, 1, H, W) 마스크 (1: 유효한 영역, 0: 무효한 영역)
    
    Returns:
        torch.Tensor: Masked Uniformity Loss 값
    """
    if valid_mask is not None:
        valid_mask = valid_mask.detach()
        valid_mask = F.interpolate(valid_mask, size=(352, 640), mode="bilinear", align_corners=False)
        valid_mask = valid_mask > 0.5  # Thresholding 적용

    losses = []
    for i in range(disp[("disp", 0)].shape[0]):
        masked_disp = disp[("disp", 0)] * valid_mask  # 마스크 내부의 깊이 값만 고려
        mean_disp = (masked_disp.sum(dim=[2, 3], keepdim=True) / (valid_mask.sum(dim=[2, 3], keepdim=True) + 1e-6))
        
        loss = torch.abs(masked_disp - mean_disp)  # 평균과의 차이를 최소화
        losses.append(loss.mean())
    loss = sum(losses) / len(losses)
    return loss



def load_pretrained_model(model, checkpoint_path="/ssd1/jm_data/depth/ssl/monodepth2/models/stereo_640x192_pretrained/", device=None):
    model["encoder"] = networks.ResnetEncoder(18, pretrained=True).to(device)
    model["depth"] = networks.DepthDecoder(model["encoder"].num_ch_enc, [0, 1, 2, 3]).to(device)

    encoder_checkpoint = torch.load(checkpoint_path+"encoder.pth", map_location=device)
    decoder_checkpoint = torch.load(checkpoint_path+"depth.pth", map_location=device)
    
    for key in ["height", "width", "use_stereo"]:
        encoder_checkpoint.pop(key, None)  # 불필요한 키 제거
    
    model["encoder"].load_state_dict(encoder_checkpoint)
    model["depth"].load_state_dict(decoder_checkpoint)
    return model
###

class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)
        self.silog_loss = SiLogLoss()
        self.mask_smooth_loss = MaskSmoothLoss()
        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.pretrained_models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.pretrained_models = load_pretrained_model(self.pretrained_models, device=self.device)
        
        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        if self.opt.use_depth_hints:
            assert 's' in self.opt.frame_ids, "Can't use depth hints without training from stereo" \
                                              "images - either add --use_stereo or remove " \
                                              "--use_depth_hints."

        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)

                self.models["pose_encoder"].to(self.device)
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "shared":
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)

            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())

        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset,
                         "kitti_depth": datasets.KITTIDepthDataset,
                         "jbnu_stereo": datasets.JBNUDepthDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")
        g_fpath = os.path.join(os.path.dirname(__file__), "glass_splits", self.opt.glass_split, "{}_files.txt")
        
        train_filenames = readlines(fpath.format("train"))
        train_glass_filenames = readlines(g_fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        val_glass_filenames = readlines(g_fpath.format("val"))
        
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, self.opt.use_depth_hints, self.opt.use_glass_hints, 
            self.opt.depth_hint_path, self.opt.glass_hint_path,
            is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, self.opt.use_depth_hints, self.opt.use_glass_hints,
            self.opt.depth_hint_path, self.opt.glass_hint_path,
            is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)


        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
            # print(key)
        if self.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            # print(inputs["color_aug", 0, 0].shape)
            # print(inputs["color_glass_aug", 0, 0].shape)
            # print(torch.unique(inputs[("color", 0, 0)]))

            features = self.models["encoder"](inputs["color_aug", 0, 0])
            features_g = None
            outputs = self.models["depth"](features)
            # print(outputs)
            outputs_g = None
            if torch.max(inputs[("color_glass_aug", 0, 0)]).item() > 0:
                features_g = self.pretrained_models["encoder"](inputs["color_glass_aug", 0, 0])
                outputs_g = self.pretrained_models["depth"](features_g)
            # print((outputs_g))
            

        if self.opt.use_glass_hints and outputs_g is not None:
            glass_mask = inputs["glass_mask"]
            # print(torch.unique(glass_mask))
            # print(glass_mask.shape)
            self.outputs = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in outputs.items()}
            self.outputs_g = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in outputs_g.items()}

            for (key, value), (key_g, value_g), scale in zip(outputs.items(), outputs_g.items(), self.opt.scales):
                # print(key)
                # print(value)
                # print()
                # print(key_g)
                # print(value_g)
                # print()
                
                scale = 3 - scale
                pred_disp = value
                pred_disp_g = value_g
                glass_mask = torch.nn.functional.interpolate(glass_mask, size=(352 // 2 ** scale, 640 // 2 ** scale), mode="bilinear", align_corners=False)
                # print(pred_disp.shape)
                # print(pred_disp_g.shape)
                # print(glass_mask.shape)
                # print(torch.unique(glass_mask))
                # print(glass_mask.shape[2] // 2 ** scale)

                pred_disp_replaced = pred_disp * (glass_mask == 0) + pred_disp_g * (glass_mask > 0)
                outputs[key] = pred_disp_replaced
            
        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))
        
        self.generate_images_pred(inputs, outputs, outputs_g)
        losses = self.compute_losses(inputs, outputs)
        # print(losses)

        if outputs_g is not None:
            silogloss = self.silog_loss(outputs, outputs_g, inputs["glass_mask"])
            l2Loss = l2_loss(outputs, outputs_g, inputs["glass_mask"])
            l1Loss = l1_loss(outputs, outputs_g, inputs["glass_mask"])
            # print(type(outputs), type(inputs['glass_mask']))
            # masked_uniform_loss = self.mask_smooth_loss(outputs, inputs["glass_mask"])
            
            # print(silogloss)
            # print(l2Loss * 0.5 + l1Loss * 0.5 + silogloss)
            # print(masked_uniform_loss)
            # print()
            losses['loss'] = losses['loss']
            
        return outputs, losses

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = next(self.val_iter)

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs, outputs_g):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]

            if outputs_g is not None:
                disp_g = outputs_g[("disp", scale)]
            else:
                disp_g = torch.zeros_like(disp)
                
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                disp_g = F.interpolate(
                    disp_g, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            _, depth_g = disp_to_depth(disp_g, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth
            self.outputs_g[("depth_g", 0, scale)] = depth_g
            
            
            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

                if self.opt.use_depth_hints:
                    if self.opt.v1_multiscale:
                        raise NotImplementedError("Depth hints are currently not implemented for v1"
                                                  "multiscale, please remove --v1_multiscape flag ")

                    elif frame_id == 's' and scale == 0:
                        # generate depth hint warped image (only max scale and for stereo image)
                        depth = inputs['depth_hint']
                        cam_points = self.backproject_depth[source_scale](
                            depth, inputs[("inv_K", source_scale)])
                        pix_coords = self.project_3d[source_scale](
                            cam_points, inputs[("K", source_scale)], T)
                        # print(f"depth hint cam_points : {cam_points}")
                        # print(f"depth hint pix_coords : {pix_coords}")
                        outputs[("color_depth_hint", frame_id, scale)] = F.grid_sample(
                            inputs[("color", frame_id, source_scale)],
                            pix_coords, padding_mode="border")
                ### +
                if self.opt.use_glass_hints:
                    if self.opt.v1_multiscale:
                        raise NotImplementedError("Glass hints are currently not implemented for v1"
                                                  "multiscale, please remove --v1_multiscape flag ")

                    elif frame_id == 's' and scale == 0:
                        # generate glass hint warped image (only max scale and for stereo image)
                        # depth = inputs['glass_hint']
                        depth = self.outputs_g[('depth_g', 0, scale)] # Output_g를 glass_hint로 하니까 이게 virtual gt로 사용됨 
                        # print(torch.unique(depth))
                        cam_points = self.backproject_depth[source_scale](
                            depth, inputs[("inv_K", source_scale)])
                        pix_coords = self.project_3d[source_scale](
                            cam_points, inputs[("K", source_scale)], T)
                        # print(f"glass hint cam_points : {cam_points}")
                        # print(f"glass hint pix_coords : {pix_coords}\n")

                        outputs[("color_glass_hint", frame_id, scale)] = F.grid_sample(
                            inputs[("color_glass", frame_id, source_scale)],
                            pix_coords, padding_mode="border")
                ###

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    @staticmethod
    def compute_proxy_supervised_loss(pred, target, valid_pixels, loss_mask):
        """ Compute proxy supervised loss (depth hint loss) for prediction.

            - valid_pixels is a mask of valid depth hint pixels (i.e. non-zero depth values).
            - loss_mask is a mask of where to apply the proxy supervision (i.e. the depth hint gave
            the smallest reprojection error)"""

        # first compute proxy supervised loss for all valid pixels
        depth_hint_loss = torch.log(torch.abs(target - pred) + 1) * valid_pixels

        # only keep pixels where depth hints reprojection loss is smallest
        depth_hint_loss = depth_hint_loss * loss_mask

        return depth_hint_loss

    @staticmethod
    def compute_loss_masks(reprojection_loss, identity_reprojection_loss,
                           depth_hint_reprojection_loss, glass_hint_reprojection_loss, glass_mask):
        """ Compute loss masks for each of standard reprojection and depth hint
        reprojection.

        identity_reprojections_loss and/or depth_hint_reprojection_loss can be None"""

        if identity_reprojection_loss is None:
            # we are not using automasking - standard reprojection loss applied to all pixels
            reprojection_loss_mask = torch.ones_like(reprojection_loss)

            if depth_hint_reprojection_loss and glass_hint_reprojection_loss:
                all_losses = torch.cat([reprojection_loss, depth_hint_reprojection_loss, glass_hint_reprojection_loss], dim=1)
                idxs = torch.argmin(all_losses, dim=1, keepdim=True)
                depth_hint_loss_mask = (idxs == 1).float()
                glass_hint_loss_mask = (idxs == 2).float()
                # glass_hint_loss_mask = glass_hint_reprojection_loss
        else:   
            # we are using automasking
            if depth_hint_reprojection_loss is not None or glass_hint_reprojection_loss is not None:
                all_losses = torch.cat([reprojection_loss, identity_reprojection_loss,
                                        depth_hint_reprojection_loss, glass_hint_reprojection_loss], dim=1)
            else:
                all_losses = torch.cat([reprojection_loss, identity_reprojection_loss], dim=1)

            idxs = torch.argmin(all_losses, dim=1, keepdim=True)

            reprojection_loss_mask = (idxs != 1).float()  # automask has index '1'
            depth_hint_loss_mask = (idxs == 2).float()  # will be zeros if not using depth hints
            glass_hint_loss_mask = (idxs == 3).float()
            # glass_hint_loss_mask = glass_hint_reprojection_loss

            # print(torch.unique(glass_hint_reprojection_loss))
        # just set depth hint mask to None if not using depth hints
        depth_hint_loss_mask = \
            None if depth_hint_reprojection_loss is None else depth_hint_loss_mask
        glass_hint_loss_mask = \
            None if glass_hint_reprojection_loss is None else glass_hint_loss_mask
        
        return reprojection_loss_mask, depth_hint_loss_mask, glass_hint_loss_mask


    def compute_losses(self, inputs, outputs):
        """Compute the reprojection, smoothness and proxy supervised losses for a minibatch
        """
        losses = {}
        total_loss = 0

        # compute depth hint reprojection loss
        if self.opt.use_depth_hints and self.opt.use_glass_hints:
            pred = outputs[("color_depth_hint", 's', 0)]
            pred_g = outputs[("color_glass_hint", 's', 0)]
            
            target = self.outputs_g[('depth_g', 0, 0)]
            # print(torch.unique(target))
            inputs['glass_hint'] = target

            inputs['glass_mask'] = F.interpolate(inputs['glass_mask'], (352, 640), mode="bilinear", align_corners=False)
            inputs['glass_hint'] *= (inputs['glass_mask'] > 0)

            # glass_mask = F.interpolate(inputs['glass_mask'], (352, 640), mode="bilinear", align_corners=False)
            inputs['glass_hint_mask'] = (inputs['glass_hint'] > 0).float() 

            depth_hint_reproj_loss = self.compute_reprojection_loss(pred, inputs[("color", 0, 0)])
            glass_hint_reproj_loss = self.compute_reprojection_loss(pred_g, inputs[("color", 0, 0)])

            # print(torch.unique(glass_hint_reproj_loss))

            # set loss for missing pixels to be high so they are never chosen as minimum
            depth_hint_reproj_loss += 1000 * (1 - inputs['depth_hint_mask'])
            glass_hint_reproj_loss += 1000 * (1 - inputs['glass_hint_mask']) # glass mask
            # print(f"depth_hint_mask : {torch.unique(inputs['depth_hint_mask'])}")
            # print(f"glass_mask : {torch.unique(inputs['glass_mask'])}\n")

        else:
            depth_hint_reproj_loss = None
            glass_hint_reproj_loss = None
            
        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # differently to Monodepth2, compute mins as we go
                    identity_reprojection_loss, _ = torch.min(identity_reprojection_losses, dim=1,
                                                              keepdim=True)
            else:
                identity_reprojection_loss = None
                if self.opt.predictive_mask:
                    # use the predicted mask
                    mask = outputs["predictive_mask"]["disp", scale]
                    if not self.opt.v1_multiscale:
                        mask = F.interpolate(
                            mask, [self.opt.height, self.opt.width],
                            mode="bilinear", align_corners=False)

                    reprojection_losses *= mask

                    # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                    weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                    loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                # differently to Monodepth2, compute mins as we go
                reprojection_loss, _ = torch.min(reprojection_losses, dim=1, keepdim=True)

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).cuda() * 0.00001

            # find minimum losses from [reprojection, identity, depth hints reprojection]
            reprojection_loss_mask, depth_hint_loss_mask, glass_hint_loss_mask = \
                self.compute_loss_masks(reprojection_loss,
                                        identity_reprojection_loss,
                                        depth_hint_reproj_loss,
                                        glass_hint_reproj_loss, 
                                        inputs['glass_mask'])

            # standard reprojection loss
            reprojection_loss = reprojection_loss * reprojection_loss_mask
            reprojection_loss = reprojection_loss.sum() / (reprojection_loss_mask.sum() + 1e-7)

            outputs["identity_selection/{}".format(scale)] = (1 - reprojection_loss_mask).float()
            losses['reproj_loss/{}'.format(scale)] = reprojection_loss

            # proxy supervision loss
            depth_hint_loss = 0
            glass_hint_loss = 0
            if self.opt.use_depth_hints:
                target = inputs['depth_hint']
                pred = outputs[('depth', 0, scale)]
                valid_pixels = inputs['depth_hint_mask']

                depth_hint_loss = self.compute_proxy_supervised_loss(pred, target, valid_pixels,
                                                                     depth_hint_loss_mask)
                depth_hint_loss = depth_hint_loss.sum() / (depth_hint_loss_mask.sum() + 1e-7)
                # save for logging
                outputs["depth_hint_pixels/{}".format(scale)] = depth_hint_loss_mask
                losses['depth_hint_loss/{}'.format(scale)] = depth_hint_loss

            ### + 
            if self.opt.use_glass_hints:
                # target = inputs['glass_hint']
                # print(self.outputs_g.keys())
                # inputs['glass_hint_mask'] *= (inputs['glass_mask'] > 0)
                target = self.outputs_g[('depth_g', 0, 0)]
                pred = outputs[('depth', 0, scale)]
                valid_pixels = inputs['glass_hint_mask']
                
                glass_hint_loss = self.compute_proxy_supervised_loss(pred, target, valid_pixels,
                                                                     glass_hint_loss_mask)
                glass_hint_loss = glass_hint_loss.sum() / (glass_hint_loss_mask.sum() + 1e-7)
                outputs["glass_hint_pixels/{}".format(scale)] = glass_hint_loss_mask
                losses['glass_hint_loss/{}'.format(scale)] = glass_hint_loss
                # print(f"glass hint loss : {glass_hint_loss}")
            ###
            loss += reprojection_loss + depth_hint_loss * 0.5 + glass_hint_loss * 1.5 # 0.5는 추가한 것

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss

        return losses


    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [360, 640], mode="bilinear", align_corners=False), 1e-3, 40)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        # mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=40)

        # print(f"pred : {torch.unique(depth_pred)}")
        # print(f"gt : {torch.unique(depth_gt)}")


        depth_errors = compute_depth_errors(depth_gt, depth_pred)
        # print(self.depth_metric_names)
        # print(depth_errors)
        # print()
        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    
                    writer.add_image(
                        "color_glass_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color_glass", frame_id, s)][j].data, self.step)

                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                ### +
                writer.add_image(
                    "disp_origin_{}/{}".format(s, j),
                    normalize_image(self.outputs[("disp", s)][j]), self.step)

                writer.add_image(
                    "disp_glass_{}/{}".format(s, j),
                    normalize_image(self.outputs_g[("disp", s)][j]), self.step)
                ### 
                                
                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)
                elif not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...].squeeze(0), self.step)

                # depth hint logging
                if self.opt.use_depth_hints:
                    if s == 0:
                        disp = 1 / (inputs['depth_hint'] + 1e-7) * inputs['depth_hint_mask']
                        writer.add_image(
                            "depth_hint/{}".format(j),
                            normalize_image(disp[j]), self.step)

                        writer.add_image(
                            "depth_hint_pixels_{}/{}".format(s, j),
                            outputs["depth_hint_pixels/{}".format(s)][j][None, ...].squeeze(0), self.step)
                        
                        writer.add_image(
                            "glass_mask{}/{}".format(s, j),
                            inputs["glass_mask"][j], self.step)
                        
                        writer.add_image(
                            "outputs(color_depth_hint_pred)/{}".format(j),
                            outputs[("color_depth_hint", 's', 0)][j], self.step)
                        
                        writer.add_image(
                            "outputs(color_glass_hint_pred)/{}".format(j),
                            outputs[("color_glass_hint", 's', 0)][j], self.step)
                        
                        # outputs[("color_depth_hint", 's', 0)]
                # glass hint logging
                if self.opt.use_glass_hints:
                    if s == 0:
                        disp = 1 / (self.outputs_g[('depth_g', 0, s)] + 1e-7) * inputs['glass_hint_mask'] # glass_hint_mask
                        writer.add_image(
                            "glass_hint/{}".format(j),
                            normalize_image(disp[j]), self.step)

                        writer.add_image(
                            "glass_hint_pixels_{}/{}".format(s, j),
                            outputs["glass_hint_pixels/{}".format(s)][j][None, ...].squeeze(0), self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized"), 
