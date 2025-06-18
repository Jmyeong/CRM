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
from padding import CRM, check_consistency
from tqdm import tqdm
import time

class silog_loss(nn.Module):
    def __init__(self, variance_focus):
        super(silog_loss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt):
        d = torch.log(depth_est) - torch.log(depth_gt)
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0

def l2_loss(pred, target, mask=None):
    if mask is not None:
        diff = (pred - target)[mask]
    else:
        diff = pred - target
    return torch.mean(diff ** 2)


class Trainer:
    def __init__(self, options):

        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)
        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models_teacher = {}
        self.models_student = {}
        self.parameters_to_train = []
        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
        # self.conv = Conv().to(self.device)
        self.silogloss = silog_loss(0.85).to(self.device)

        K = np.array([[263.9025, 0, 323.5725],
            [0, 263.675, 179.957],
            [0, 0, 1]
            ], dtype=np.float32)
        self.crm = CRM(K).to(self.device)
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

                    
        self.models_teacher["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models_student["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models_teacher["encoder"].to(self.device)
        self.models_student["encoder"].to(self.device)
        self.parameters_to_train += list(self.models_student["encoder"].parameters())

        self.models_teacher["depth"] = networks.DepthDecoder(
            self.models_teacher["encoder"].num_ch_enc, self.opt.scales)
        self.models_student["depth"] = networks.DepthDecoder(
            self.models_student["encoder"].num_ch_enc, self.opt.scales)
        self.models_teacher["depth"].to(self.device)
        self.models_student["depth"].to(self.device)
        self.parameters_to_train += list(self.models_student["depth"].parameters())

        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models_teacher["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)
                self.models_student["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)

                self.models_teacher["pose_encoder"].to(self.device)
                self.models_student["pose_encoder"].to(self.device)
                self.parameters_to_train += list(self.models_student["pose_encoder"].parameters())

                self.models_teacher["pose"] = networks.PoseDecoder(
                    self.models_teacher["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)
                self.models_student["pose"] = networks.PoseDecoder(
                    self.models_student["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "shared":
                self.models_teacher["pose"] = networks.PoseDecoder(
                    self.models_teacher["encoder"].num_ch_enc, self.num_pose_frames)
                self.models_student["pose"] = networks.PoseDecoder(
                    self.models_student["encoder"].num_ch_enc, self.num_pose_frames)

            elif self.opt.pose_model_type == "posecnn":
                self.models_teacher["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)
                self.models_student["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            self.models_teacher["pose"].to(self.device)
            self.models_student["pose"].to(self.device)
            self.parameters_to_train += list(self.models_student["pose"].parameters())

        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            self.models_teacher["predictive_mask"] = networks.DepthDecoder(
                self.models_teacher["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models_student["predictive_mask"] = networks.DepthDecoder(
                self.models_student["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models_teacher["predictive_mask"].to(self.device)
            self.models_student["predictive_mask"].to(self.device)

            self.parameters_to_train += list(self.models_student["predictive_mask"].parameters())

        self.model_optimizer = optim.AdamW(self.parameters_to_train, self.opt.learning_rate, weight_decay=1e-2, eps=1e-6)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()
        # Load teacher checkpoitns
        encoder_path = os.path.join(self.opt.teacher_path, "encoder.pth")
        decoder_path = os.path.join(self.opt.teacher_path, "depth.pth")
        checkpoint_encoder = torch.load(encoder_path)
        checkpoint_decoder = torch.load(decoder_path)
        encoder_ckpt = torch.load(encoder_path)
        filtered_encoder_dict = {k: v for k, v in encoder_ckpt.items()
                                if k.startswith("encoder.") and k not in ["height", "width", "use_stereo"]}
        model_dict = self.models_teacher["encoder"].state_dict()
        filtered_encoder_dict = {k: v for k, v in filtered_encoder_dict.items() if k in model_dict}
        self.models_teacher["encoder"].load_state_dict(filtered_encoder_dict)
        self.models_teacher["depth"].load_state_dict(checkpoint_decoder)

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

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, self.opt.use_depth_hints, self.opt.depth_hint_path,
            is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, self.opt.use_depth_hints, self.opt.depth_hint_path,
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
        for m in self.models_student.values():
            m.train()
        for m in self.models_teacher.values():
            m.eval()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models_student.values():
            m.eval()
        for m in self.models_teacher.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            print(f"Epoch {self.epoch} done.")
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(tqdm(self.train_loader)):

            before_op_time = time.time()

            outputs, pseudo_gt, losses = self.process_batch(inputs)
            
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

                self.log("train", inputs, pseudo_gt, outputs, losses)
                self.val()

            self.step += 1

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        self.models_teacher["encoder"].eval()
        self.models_teacher["depth"].eval()

        losses = {}
        for key, ipt in inputs.items():
            if isinstance(ipt, torch.Tensor):
                inputs[key] = ipt.to(self.device)


        if self.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models_teacher["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            features_T = {}
            for i, k in enumerate(self.opt.frame_ids):
                features_T[k] = [f[i] for f in all_features]
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            # features_T = self.models_teacher["encoder"](inputs["color_aug", 0, 0])
            # pseudo_gt = self.models_teacher["depth"](features_T) # scale 0 ~ 3
            pseudo_gt = {}
            features_S = self.models_student["encoder"](inputs["color_aug", 0, 0])
            outputs = self.models_student["depth"](features_S)
  
        total_loss = 0
        
        do_crm = True
        for scale in range(self.num_scales):
            disp = outputs[("disp", scale)]
            glass_mask = inputs["mask"]
            # print(torch.unique(glass_mask))
            inputs["mask", scale] = F.interpolate(glass_mask, [disp.shape[2], disp.shape[3]], mode='bilinear', align_corners=False)
            inputs["mask", scale] = torch.where(inputs["mask", scale] > 0.01, 1.0, 0.0)

            processed = F.interpolate(inputs["depth_gt"], [disp.shape[2], disp.shape[3]], mode='bilinear', align_corners=False)
            
            if do_crm:
                pseudo_gt[("disp", scale)] = processed
                gt = processed
                
                l2_valid_mask = inputs["mask", scale]
                l2_valid_mask = l2_valid_mask.bool()
                l2loss = l2_loss(disp[l2_valid_mask], gt[l2_valid_mask]) * 20.0
                print(l2loss)

            else:
                gt = pseudo_gt[("disp", scale)]
                l2loss = 0.0
                
            mask = gt > 0

            valid_mask = mask & (disp > 0)
            _, pred_depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = pred_depth
            loss = self.silogloss(disp[valid_mask], gt[valid_mask])
            # print(valid_mask.shape, inputs["mask", scale].shape)
            # print(torch.unique(inputs["mask", scale]))
            # print(disp.shape, gt.shape)
            total_loss += (loss + l2loss)
            losses[("loss", scale)] = (loss + l2loss)

        losses["loss"] = total_loss / self.num_scales

        return outputs, pseudo_gt, losses

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
            outputs, pseudo_gt, losses = self.process_batch(inputs)
            print(losses)
            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, pseudo_gt, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [360, 640], mode="bilinear", align_corners=False), 1e-3, 50)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt_for_valid"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        # mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=50)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)
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

    def log(self, mode, inputs, pseudo_gt, outputs, losses):
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
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)
                writer.add_image(
                    "depth_{}/{}".format(s, j),
                    normalize_image(outputs[('depth', 0, s)][j]), self.step)
                writer.add_image(
                    "gt_{}/{}".format(s, j),
                    normalize_image(pseudo_gt[("disp", s)][j]), self.step)
                writer.add_image(
                    "gt_for_valid_{}/{}".format(s, j),
                    normalize_image(inputs[("depth_gt_for_valid")][j]), self.step)

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

        for model_name, model in self.models_student.items():
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
            model_dict = self.models_student[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models_student[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
