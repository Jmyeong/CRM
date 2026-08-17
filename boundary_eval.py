from __future__ import absolute_import, division, print_function
import argparse
import sys

import os
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import datasets
import networks
from utils import normalize_image
from torch.utils.tensorboard.writer import SummaryWriter
from PIL import Image
from tqdm import tqdm

project_root = os.path.join(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(project_root, "utils"))
from image import *
sys.path.append(os.path.join(project_root, "crm"))
from module import *

class RunEvaluate():
    def __init__(self, mode="total", is_mask=True, in_mask=True, opt=None):
        super(RunEvaluate, self).__init__()
        self.opt = opt
        self.writers = {}
        self.project_root = project_root
        self.splits_dir = os.path.join(os.path.dirname(__file__), "splits")
        self.mode = mode
        self.is_mask = is_mask
        self.in_mask = in_mask
        self.STEREO_SCALE_FACTOR = 1.2 # convert this value to 5.4 if using kitti dataset
        self.K = np.array([[260.8747863769531, 0, 321.9953308105469],
            [0, 260.8747863769531, 179.68511962890625],
            [0, 0, 1]
            ], dtype=np.float32)
        self.K = to_tensor_(self.K)
        
    def compute_errors(self, gt, pred):
        if isinstance(gt, list) or gt.dtype == 'O':
            gt = np.array(gt, dtype=np.float64)
        if isinstance(pred, list) or pred.dtype == 'O':
            pred = np.array(pred, dtype=np.float64)
            
        thresh = np.maximum((gt / pred) , (pred / gt))
        
        a1 = (thresh < 1.25).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()
        
        rmse = np.sqrt(((gt - pred) ** 2).mean())
        rmse_log = (np.log(gt) - np.log(pred)) ** 2
        rmse_log = np.sqrt(rmse_log.mean())
        
        abs_rel = np.mean(np.abs(gt - pred) / gt)

        sq_rel = np.mean(((gt - pred) ** 2) / gt)
        
        return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
    
    def batch_post_process_disparity(self, l_disp, r_disp):
        _, h, w = l_disp.shape
        m_disp = 0.5 * (l_disp + r_disp)
        l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
        l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
        r_mask = l_mask[:, :, ::-1]
        return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp
    
    def evaluate(self):
        MIN_DEPTH = 1e-3
        MAX_DEPTH = 50
        
        assert sum((self.opt.eval_mono, self.opt.eval_stereo)) == 1, \
            "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"
        print("-> Loading weights from {}".format(self.opt.load_weights_folder))
        if self.mode == "total":
            filenames = readlines(os.path.join(self.splits_dir, self.opt.eval_split, "test_files.txt"))
        elif self.mode == "tom":
            filenames = readlines(os.path.join(self.splits_dir, self.opt.eval_split, "transp_files.txt"))
        else:
            ValueError(f"Unknown mode: {self.mode}")
        
        encoder_path = os.path.join(self.opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(self.opt.load_weights_folder, "depth.pth")
        
        encoder_dict = torch.load(encoder_path)
        if self.opt.backbone in ["resnet", "resnet_lite"]:
            encoder = networks.ResnetEncoderDecoder(num_layers=self.opt.num_layers, num_features=self.opt.num_features, model_dim=self.opt.model_dim)
        elif self.opt.backbone == "resnet18_lite":
            encoder = networks.LiteResnetEncoderDecoder(model_dim=self.opt.model_dim)
        elif self.opt.backbone == "eff_b5":
            encoder = networks.BaseEncoder.build(num_features=self.opt.num_features, model_dim=self.opt.model_dim)
        else: 
            encoder = networks.Unet(pretrained=(not self.opt.load_pretrained_model), backbone=self.opt.backbone, in_channels=3, num_classes=self.opt.model_dim, decoder_channels=self.opt.dec_channels)

        if self.opt.backbone.endswith("_lite"):
            depth_decoder = networks.Lite_Depth_Decoder_QueryTr(in_channels=self.opt.model_dim, patch_size=self.opt.patch_size, dim_out=self.opt.dim_out, embedding_dim=self.opt.model_dim, 
                                                        query_nums=self.opt.query_nums, num_heads=4, min_val=self.opt.min_depth, max_val=self.opt.max_depth)
        else:
            depth_decoder = networks.Depth_Decoder_QueryTr(in_channels=self.opt.model_dim, patch_size=self.opt.patch_size, dim_out=self.opt.dim_out, embedding_dim=self.opt.model_dim, 
                                                   query_nums=self.opt.query_nums, num_heads=4, min_val=self.opt.min_depth, max_val=self.opt.max_depth)
        
        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path))

        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()
        
        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))
        
        if self.opt.eval_split == "jbnu_stereo":
            dataset = datasets.JBNUDepthDataset(self.opt.data_path, filenames, encoder_dict['height'], encoder_dict['width'], [0], 1, is_train=False)
        else:
            dataset = datasets.KITTIRAWDataset(self.opt.data_path, filenames, encoder_dict['height'], encoder_dict['width'], [0], 1, is_train=False)
        dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=self.opt.num_workers, pin_memory=True, drop_last=False)
            
        step = 0
        ratios = []
        errors = []
        with torch.no_grad():
            for i, data in enumerate(tqdm(dataloader)):
                step = step + 1
                input_color = data[("color", 0, 0)].cuda()
                
                output = depth_decoder(encoder(input_color))
                pred_disp = output[("disp", 0)]
                pred_depth = pred_disp
                pred_depth = F.interpolate(pred_depth, (360, 640), mode="bilinear", align_corners=False)
                
                # print(pred_depth.squeeze(0).shape)
                # print(self.K.squeeze(0).squeeze(0).shape)
                consistency_mask = check_consistency(
                    pred_depth.squeeze(0),
                    self.K.squeeze(0).squeeze(0),
                    baseline=0.12,
                    z_thresh=0.05
                )
                consistency_mask = squeezing(to_numpy(consistency_mask * 255.0))
                
                pred_depth = to_numpy(pred_depth)
                pred_depth = squeezing(pred_depth)
                

                gt_depth = data["depth_gt_transp"]
                gt_depth = squeezing(to_numpy(gt_depth))
                # vis_imgs(pred_depth=pred_depth, gt_depth=gt_depth, consistency_mask=consistency_mask)
                gt_height, gt_width = gt_depth.shape

                if self.opt.eval_split == "eigen":
                    mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

                    crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                                    0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
                    crop_mask = np.zeros(mask.shape)
                    crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
                    mask = np.logical_and(mask, crop_mask)

                else:
                    mask = gt_depth > 0
                    
                mask = np.logical_and(mask, consistency_mask)
                
                transp_mask = data["mask"]
                transp_mask = squeezing(to_numpy(transp_mask))
                
                if self.is_mask:
                    if not self.in_mask:
                        out_mask = 255 - transp_mask
                        combined_mask = np.logical_and(mask, out_mask)
                    else:
                        combined_mask = np.logical_and(mask, transp_mask)
                    pred_depth = pred_depth[combined_mask]
                    gt_depth = gt_depth[combined_mask]
                else:
                    pred_depth = pred_depth[mask]
                    gt_depth = gt_depth[mask]

                pred_depth *= self.opt.pred_depth_scale_factor
                if not self.opt.disable_median_scaling:
                    ratio = np.median(gt_depth) / np.median(pred_depth)
                    ratios.append(ratio)
                    pred_depth *= ratio
                
                pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
                pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
                errors.append(self.compute_errors(gt_depth, pred_depth))
                
                
        if not opt.disable_median_scaling:
            ratios = np.array(ratios)
            med = np.median(ratios)
            print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))
            
        mean_errors = np.array(errors).mean(0)

        print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
        print("\n-> Done!")

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)
        
if __name__ == "__main__":
    options = MonodepthOptions()
    options.parser.convert_arg_line_to_args = convert_arg_line_to_args
    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        opt = options.parser.parse_args([arg_filename_with_prefix])
    else:
        opt = options.parser.parse_args()
    run_eval = RunEvaluate(mode="total", is_mask=False, in_mask=False, opt=opt)
    run_eval.evaluate()
    
        