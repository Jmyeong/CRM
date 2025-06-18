import torch
import torchvision
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

import networks

from torchvision import transforms

import tools
from PIL import Image
from glob import glob
import argparse

import matplotlib.pyplot as plt
import cv2

import os
from tqdm import tqdm
from padding import CRM

drive = 4
image_num = 2
date = "2025_05_08"
test = f"{date}_drive_000{drive}_sync"

K = np.array([[260.8747863769531, 0, 321.9953308105469],
            [0, 260.8747863769531, 179.68511962890625],
            [0, 0, 1]
            ], dtype=np.float32)
crm = CRM(K).to("cuda")

def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction"""
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

def predict(model, image, name, output_type, date, test, image_num):
    predicts = {}
    predicts["depth"] = []

    features = model["encoder"](image)
    predicts["disparity"] = model["depth"](features)
    
    # KeyError 방지: 딕셔너리 키 접근 수정
    predicts["disparity"]["disp", 0] = F.interpolate(
        predicts["disparity"]["disp", 0], 
        size=(args.img_height, args.img_width), 
        mode="bilinear", 
        align_corners=False
    )
    # _, depth = disp_to_depth(predicts["disparity"]["disp", 0], args.min_depth, args.max_depth)
    predicts["depth"] = predicts["disparity"]["disp", 0]  
    depth_numpy = predicts["depth"].squeeze(0).squeeze(0).detach().cpu().numpy()
    # print(depth_numpy.shape)

    # np.save(f"{output_path}/{name}", depth_numpy)
    return depth_numpy

def median_depth(images):
    """5개의 이미지에서 중간값을 계산"""
    images_stack = np.stack(images, axis=0)
    median_image = np.median(images_stack, axis=0)
    return median_image

def visualization(model, device, input_path, mask_path, transform, output_type, date, test, image_num):
    
    files = sorted(glob(input_path + "*.png", recursive=False))
    masks_pre = sorted(glob(mask_path + "*.png", recursive=False))
    # print(masks_pre)
    masks = []
    if os.path.exists(mask_path):
        mask_path_base = "/".join(masks_pre[0].split("/")[:-1])
    else:
        mask_path_base = None
    for i in range(len(files)):
        if os.path.exists(mask_path):
            check_masks = os.path.join(mask_path_base, files[i].split("/")[-1])
        else:
            check_masks = False
            
        if not check_masks:
            masks.append(None)
        else:
            masks.append(check_masks)
        print(check_masks)
    # 각 이미지에 대해 5개의 예측값을 저장할 리스트
    depth_images = []
    for file, mask in tqdm(list(zip(files, masks)), desc=f"{test}/image_0{image_num}"):
        name = os.path.basename(file).split(".")[0]
        image = Image.open(file).convert("RGB")
        image = transform(image).to(device)
        if mask is not None and os.path.exists(mask):
            glass_mask = cv2.imread(mask)
            glass_mask = cv2.cvtColor(glass_mask, cv2.COLOR_RGB2GRAY)
            # glass_mask = cv2.resize(glass_mask, (640, 352))
            glass_mask = (glass_mask > 50).astype(np.uint8) * 255.0
            cv2.imwrite("./mask.png", glass_mask * 255.0)
        else:
            glass_mask = np.zeros((352, 640))
        # print(np.unique(glass_mask))
        # print(glass_mask.shape)

        glass_mask = transform(Image.fromarray(glass_mask)).to(device)
        depth_image = predict(model, image.unsqueeze(0), name, output_type, date, test, image_num)
        depth_image = torch.from_numpy(depth_image).to(device)
        # depth_image = depth_image.unsqueeze(0).unsqueeze(0)
        # print(depth_image.shape)
        # depth_image = depth_image.squeeze(0).squeeze(0)
        depth_image = crm(depth_image, glass_mask)
        depth_image = depth_image.unsqueeze(0).unsqueeze(0)
        depth_image = F.interpolate(depth_image, (360, 640), mode="bilinear", align_corners=False)
        depth_image = depth_image.squeeze(0).squeeze(0)

        depth_image = depth_image.detach().cpu().numpy()
        # print(depth_image.shape)
        # print(np.unique(depth_image))
        # output_path = f"./padded_depth/{date}/{test}/image_0{image_num}"
        output_path = f"/ssd1/jm_data/depth/ssl/monodepth2/jbnu_stereo/{date}/{test}/proj_depth/groundtruth_crm/image_0{image_num}"
        # print(output_path)
        os.makedirs(output_path, exist_ok=True)
        np.save(f"{output_path}/{name}", depth_image)

    print(f"Depth maps saved for {test}/image_0{image_num}")


def main(args):
    transform = transforms.Compose([
        transforms.Resize((args.img_height, args.img_width)),
        transforms.ToTensor(),
    ])
    
    output_type = "video"
    model = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model["encoder"] = networks.ResnetEncoder(args.num_layers, pretrained=True).to(device)
    model["depth"] = networks.DepthDecoder(model["encoder"].num_ch_enc, [0, 1, 2, 3]).to(device)

    encoder_checkpoint = torch.load(f"/ssd1/jm_data/depth/ssl/depth-hints/logs/jbnu_stereo_teacher_monodepth2/models/weights_{args.epoch}/encoder.pth", map_location=device)
    decoder_checkpoint = torch.load(f"/ssd1/jm_data/depth/ssl/depth-hints/logs/jbnu_stereo_teacher_monodepth2/models/weights_{args.epoch}/depth.pth", map_location=device)

    for key in ["height", "width", "use_stereo"]:
        encoder_checkpoint.pop(key, None)

    model["encoder"].load_state_dict(encoder_checkpoint)
    model["depth"].load_state_dict(decoder_checkpoint)

    date_dir = f"/ssd1/jm_data/depth/ssl/monodepth2/jbnu_stereo/{args.date}"
    drive_paths = sorted(glob(os.path.join(date_dir, f"{args.date}_drive_0*_sync")))

    for drive_path in drive_paths:
        drive_name = os.path.basename(drive_path)
        for image_num in [2, 3]:
            input_path = os.path.join(drive_path, f"image_0{image_num}/data/")
            # mask_path = os.path.join(drive_path, f"mask/image_0{image_num}/data/")
            mask_path = os.path.join("/ssd1/jm_data/Grounded-Segment-Anything/outputs", "2025_05_08", drive_name, f"image_0{image_num}/mask/")
            # print(input_path)
            # if os.path.isdir(input_path) and os.path.isdir(mask_path):
            print(mask_path)
            print(f"Processing: {drive_name}/image_0{image_num}")
            visualization(model, device=device, input_path=input_path, 
                            mask_path=mask_path, transform=transform, 
                            output_type=output_type, date=args.date, test=drive_name, image_num=image_num)
            # else:
            #     print(f"Skipped: {drive_name}/image_0{image_num} (missing folders)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Depth Prediction")
    parser.add_argument("--num_layers", type=int, default=18, help="Number of layers in ResNet encoder")
    parser.add_argument("--pretrained", type=bool, default=True, help="Use pretrained weights")
    parser.add_argument("--img_height", type=int, default=352, help="Input image height")
    parser.add_argument("--img_width", type=int, default=640, help="Input image width")
    parser.add_argument("--min_depth", type=float, default=0.1, help="Minimum depth value")
    parser.add_argument("--max_depth", type=float, default=50.0, help="Maximum depth value")
    parser.add_argument("--epoch", type=int, default=0, help="Epoch to load weights from")
    parser.add_argument("--date", type=str, default="2025_05_08", help="Epoch to load weights from")

    args = parser.parse_args()
    main(args)
