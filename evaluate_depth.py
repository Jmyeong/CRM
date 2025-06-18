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
from PIL import Image

from tqdm import tqdm
from fvcore.nn import FlopCountAnalysis, parameter_count_table

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")
mode = "tom"
is_mask = False
in_mask = True
# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 1.2 # 5.4


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    if isinstance(gt, list) or gt.dtype == 'O':  # 'O'는 object 타입을 의미
        gt = np.array(gt, dtype=np.float64)  # float64로 변환

    # pred도 동일한 방식으로 변환
    if isinstance(pred, list) or pred.dtype == 'O':
        pred = np.array(pred, dtype=np.float64)
    
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


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 50 # 80

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))
        
        if mode == "total":
            filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        elif mode == "tom":
            filenames = readlines(os.path.join(splits_dir, opt.eval_split, "tom_files.txt"))
        
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
        # print(filenames)
        mask_path = []
        for i in filenames:
            parts = i.split("/")
            drive, index, side = parts[1].split(" ")
            path = os.path.join("/ssd1/jm_data/Grounded-Segment-Anything/outputs",  parts[0], drive, "image_03", "mask", f"{str(index).zfill(10)}.png")
            # print(path)
            mask_path.append(path)
        encoder_dict = torch.load(encoder_path)

        dataset = datasets.JBNUDepthDataset(opt.data_path, filenames,
                                           encoder_dict['height'], encoder_dict['width'],
                                           [0], 4, is_train=False, use_depth_hints=False, use_glass_hints=False)
        dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)

        encoder = networks.ResnetEncoder(opt.num_layers, False)
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path))

        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()
        
        pred_disps = []
        pred_depth_save = []
        
        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))

        with torch.no_grad():
            for data in dataloader:
                input_color = data[("color", 0, 0)].cuda()
                if opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)
                # print(input_color.shape)
                # print(encoder(input_color)[0].shape)
                output = depth_decoder(encoder(input_color))
                print(torch.unique(output[("disp", 0)]))
                pred_disp, pred_depth = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()
                print(np.unique(pred_depth.cpu()[:,0].numpy()))
                print()
                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])
                # print(pred_disp.shape)
                # print(np.unique(output[("disp", 0)].cpu()[:,0].numpy()))
                pred_depth_save.append(pred_depth.cpu()[:,0].numpy())
                pred_disps.append(pred_disp)
        pred_disps_save = np.concatenate(pred_depth_save)
        pred_disps = np.concatenate(pred_disps)

    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps_save)

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    elif opt.eval_split == 'benchmark':
        save_dir = os.path.join(opt.load_weights_folder, "benchmark_predictions")
        print("-> Saving out benchmark predictions to {}".format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(len(pred_disps)):
            disp_resized = cv2.resize(pred_disps[idx], (640, 360)) # 1216, 352
            depth = STEREO_SCALE_FACTOR / disp_resized
            depth = np.clip(depth, 0, 50) # 40
            depth = np.uint16(depth * 256)
            save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
            cv2.imwrite(save_path, depth)

        print("-> No ground truth is available for the KITTI benchmark, so not evaluating. Done.")
        quit()
    if mode == "total":
        gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    elif mode == "tom":
        gt_path = os.path.join(splits_dir, opt.eval_split, "tom_gt_depths.npz")

    # print(gt_path)
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    print("-> Evaluating")

    if opt.eval_stereo:
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []
    final_value = []
    
    for i in tqdm(range(pred_disps.shape[0])):

        gt_depth = gt_depths[i]
        # print(np.unique(gt_depth))
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i] # / 256
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        if opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        else:
            mask = gt_depth > 0

        # pred_depth = pred_depth[mask]
        # gt_depth = gt_depth[mask]
        if is_mask:
            if not in_mask:
                # print("not in_mask")
                if not os.path.exists(mask_path[i]):
                    glass_mask = np.zeros_like(mask)
                else:
                    glass_mask = Image.open(mask_path[i]).convert("L")
                    # glass_mask = transform(glass_mask.resize((640, 352))).to("cuda").unsqueeze(0)
                    glass_mask = np.array(glass_mask)
                    glass_mask = cv2.resize(glass_mask, (640, 360))
                # print(mask.shape, glass_mask.shape)
                # print(np.unique(glass_mask))
                glass_mask = (glass_mask > 127).astype(np.uint8) * 255
                if len(np.unique(glass_mask)) < 2:
                    continue

                out_mask = 255 - glass_mask
                # print(np.unique(mask))
                # print(np.unique(out_mask))
                # print()
                combined_mask = np.logical_and(mask, out_mask)
                # cv2.imwrite("./out_mask.png", out_mask)
                # print(np.unique(combined_mask))
                # print(pred_depth.shape, gt_depth.shape, glass_mask.shape, out_mask.shape)
                pred_depth = pred_depth[combined_mask]
                gt_depth = gt_depth[combined_mask]
                # print(gt_depth.shape)
                # cv2.imwrite("./gt_depth.png", gt_depth)
                # cv2.imwrite("./combined_mask.png", combined_mask.astype(np.int8) * 255.0)
            else:
                # print("in_mask")
                if not os.path.exists(mask_path[i]):
                    glass_mask = np.zeros_like(mask)
                else:
                    glass_mask = Image.open(mask_path[i]).convert("L")
                    # glass_mask = transform(glass_mask.resize((640, 352))).to("cuda").unsqueeze(0)
                    glass_mask = np.array(glass_mask)
                    glass_mask = cv2.resize(glass_mask, (640, 360))
                # print(mask.shape, glass_mask.shape)
                glass_mask = (glass_mask > 127).astype(np.uint8) * 255
                if len(np.unique(glass_mask)) < 2:
                    continue
                combined_mask = np.logical_and(mask, glass_mask)
                cv2.imwrite("./glass_mask.png", glass_mask)
                # print(np.unique(out_mask))
                # print(pred_depth.shape, gt_depth.shape, glass_mask.shape, out_mask.shape)
                pred_depth = pred_depth[combined_mask]
                gt_depth = gt_depth[combined_mask]
        else:
            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]


        # print(np.unique(gt_depth))
        pred_depth *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio
        
        # pred_depth = pred_depth / 1.15
        # print(f"pred : {np.unique(pred_depth)}")
        # print(f"gt : {np.unique(gt_depth)}")
        # print(pred_depth.shape)
        # print(gt_depth.shape)
        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
        # print(np.unique(gt_depth))
        # gt_depth = gt_depth.astype(int)
        # pred_depth = pred_depth.astype(int)
        abs_rel = compute_errors(gt_depth, pred_depth)[0]
        if abs_rel <= 0.095:
            final_value.append(abs_rel)
            print(f"satisfied image index : {i}")
        errors.append(compute_errors(gt_depth, pred_depth))
        # print(len(final_value))
    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")

    print(f"final abs_rel : {np.array(final_value).mean(0)}")

if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
