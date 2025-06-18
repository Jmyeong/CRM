from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import datasets
import networks

from tqdm import tqdm
import argparse
from PIL import Image

from torchvision import transforms as T

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")
mode = "single"
# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 1.3 # 5.4


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    if isinstance(gt, list) or gt.dtype == 'O':  # 'O'는 object 타입을 의미
        gt = np.array(gt, dtype=np.float64)  # float64로 변환

    # pred도 동일한 방식으로 변환
    if isinstance(pred, list) or pred.dtype == 'O':
        pred = np.array(pred, dtype=np.float64)
    
    # print(f"gt : {gt.dtype}")
    # print(f"pred : {pred.dtype}")
    # print(f"gt : {np.unique(gt)}")
    # print(f"pred : {np.unique(pred)}")

    # eps = 1e-6
    # pred = np.maximum(pred, eps) # inf 방지
    
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    # print(f"log(pred) : {np.log(pred)}")
    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(args):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 40 # 80

    transform = T.Compose([
        T.ToTensor()
    ])
    assert sum((args.eval_mono, args.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if args.ext_disp_to_eval is None:

        args.load_weights_folder = os.path.expanduser(args.load_weights_folder)

        assert os.path.isdir(args.load_weights_folder), \
            "Cannot find a folder at {}".format(args.load_weights_folder)

        print("-> Loading weights from {}".format(args.load_weights_folder))
        
        encoder_path = os.path.join(args.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(args.load_weights_folder, "depth.pth")

        encoder_dict = torch.load(encoder_path)

        encoder = networks.ResnetEncoder(args.num_layers, False)
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path))

        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()

        pred_disps = []

        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))

        with torch.no_grad():
            input_color = Image.open( os.path.join(args.data_path, args.date, "{}_drive_{}_sync".format(args.date, str(args.drive).zfill(4)), "image_02/data", "{}.png".format(str(args.index).zfill(10))))
            input_color = transform(input_color.resize((640, 352))).to("cuda").unsqueeze(0)
            print(input_color.shape)
            if args.post_process:
                # Post-processed results require each image to have two forward passes
                input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

            output = depth_decoder(encoder(input_color))
            
            # print(torch.unique(output[("disp", 0)]))
            pred_disp, _ = disp_to_depth(output[("disp", 0)], 1e-1, 40)
            pred_disp = pred_disp.cpu()[:, 0].numpy()
            # print(np.unique(pred_disp / 20))
            if args.post_process:
                N = pred_disp.shape[0] // 2
                pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

            pred_disps.append(pred_disp)

        pred_disps = np.concatenate(pred_disps)

    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(args.ext_disp_to_eval))
        pred_disps = np.load(args.ext_disp_to_eval)

        if args.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    if args.save_pred_disps:
        output_path = os.path.join(
            args.load_weights_folder, "disps_{}_split.npy".format(args.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    if args.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    elif args.eval_split == 'benchmark':
        save_dir = os.path.join(args.load_weights_folder, "benchmark_predictions")
        print("-> Saving out benchmark predictions to {}".format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(len(pred_disps)):
            disp_resized = cv2.resize(pred_disps[idx], (640, 360)) # 1216, 352
            depth = STEREO_SCALE_FACTOR / disp_resized
            depth = np.clip(depth, 0, 40) # 40
            depth = np.uint16(depth * 256)
            save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
            cv2.imwrite(save_path, depth)

        print("-> No ground truth is available for the KITTI benchmark, so not evaluating. Done.")
        quit()
    gt_path = os.path.join(args.data_path, args.date, "{}_drive_{}_sync".format(args.date, str(args.drive).zfill(4)), "proj_depth/gt_interpolated/image_02", "{}.npy".format(str(args.index).zfill(10)))

    print(gt_path)
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)

    print("-> Evaluating")

    if args.eval_stereo:
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        args.disable_median_scaling = True
        args.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []

    gt_depth = gt_depths

    # print(np.unique(gt_depth))
    gt_height, gt_width = gt_depth.shape[:2]
    # print(pred_disps.shape)
    pred_disp = pred_disps[0] # / 256
    pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
    # print(np.unique(pred_disp))

    pred_depth = 1 / pred_disp

    if args.eval_split == "eigen":
        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

        crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                            0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        mask = np.logical_and(mask, crop_mask)

    else:
        mask = gt_depth > 0

    if args.mask:
        glass_mask = Image.open(os.path.join(args.data_path, args.date, "{}_drive_{}_sync".format(args.date, str(args.drive).zfill(4)), "mask", "image_02/data", "{}_label.png".format(str(args.index).zfill(10)))).convert("L")
        # glass_mask = transform(glass_mask.resize((640, 352))).to("cuda").unsqueeze(0)
        glass_mask = np.array(glass_mask)
        out_mask = 255 - glass_mask
        combined_mask = np.logical_and(mask, out_mask)

        print(np.unique(out_mask))
        print(pred_depth.shape, gt_depth.shape, glass_mask.shape, out_mask.shape)
        pred_depth = pred_depth[combined_mask]
        gt_depth = gt_depth[combined_mask]
        # cv2.imwrite("./gt_depth.png", gt_depth)
        # cv2.imwrite("./out_mask.png", out_mask)

    else:
        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]
    # print(np.unique(gt_depth))
    pred_depth *= args.pred_depth_scale_factor
    if not args.disable_median_scaling:
        ratio = np.median(gt_depth) / np.median(pred_depth)
        ratios.append(ratio)
        pred_depth *= ratio
        
    print(f"pred : {np.unique(pred_depth)}")
    print(f"gt : {np.unique(gt_depth)}")
    # print(pred_depth.shape)
    # print(gt_depth.shape)
    pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
    pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
    # print(np.unique(gt_depth))
    # gt_depth = gt_depth.astype(int)
    # pred_depth = pred_depth.astype(int)
    # cv2.imwrite("./pred_output.png", pred_depth)
    errors.append(compute_errors(gt_depth, pred_depth))

    if not args.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default="2025_02_28")
    parser.add_argument('--drive', type=int, default=2)
    parser.add_argument('--index', type=int, default=2405)

    # 필요한 추가 인자들
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--split', type=str, default='jbnu_stereo')
    parser.add_argument('--dataset', type=str, default='jbnu_stereo')
    parser.add_argument('--png', action='store_true')
    parser.add_argument('--eval_stereo', action='store_true')
    parser.add_argument('--eval_mono', action='store_true')
    parser.add_argument('--eval_split', type=str, default='jbnu_stereo')
    parser.add_argument('--save_pred_disps', action='store_true')
    parser.add_argument('--post_process', action='store_true')
    parser.add_argument('--load_weights_folder', type=str, required=True)
    parser.add_argument('--ext_disp_to_eval', type=str, default=None)
    parser.add_argument('--no_eval', action='store_true')
    parser.add_argument('--disable_median_scaling', action='store_true')
    parser.add_argument('--pred_depth_scale_factor', type=float, default=1)
    parser.add_argument('--eval_eigen_to_benchmark', action='store_true')
    parser.add_argument('--num_layers', type=int, default=18)
    parser.add_argument('--mask', action='store_true')


    args = parser.parse_args()
    evaluate(args)
