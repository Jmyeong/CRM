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

drive = 2
image_num = 2
date = "2025_03_23"
test = f"{date}_drive_000{drive}_sync"

def inpaint_color(color, mask, N=5):
    # 랜덤 색상 생성 (B, G, R)
    fill_color = np.random.randint(0, 256, 3, dtype=np.uint8)
    
    # 원본 이미지 복사 후 마스크 영역에 랜덤 색 적용
    inpainted_color = np.array(color.copy())
    inpainted_color[mask > 0] = fill_color
    inpainted_color = Image.fromarray(inpainted_color)
    return inpainted_color

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
    _, depth = disp_to_depth(predicts["disparity"]["disp", 0], args.min_depth, args.max_depth)
    predicts["depth"] = depth  
    depth_numpy = depth.squeeze(0).squeeze(0).detach().cpu().numpy()
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
    masks = sorted(glob(mask_path + "*.png", recursive=False))
    
    # 각 이미지에 대해 5개의 예측값을 저장할 리스트
    depth_images = []

    for file, mask in tqdm(list(zip(files, masks)), desc=f"{test}/image_0{image_num}"):
        name = os.path.basename(file).split(".")[0]
        image = Image.open(file).convert("RGB")
        image = np.array(image)
        glass_mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
        
        # 5개의 inpainted 이미지를 생성
        inpainted_images = []
        for _ in range(5):
            inpainted_image = inpaint_color(image, glass_mask)
            # inpaint된 이미지를 배치에 추가
            inpainted_images.append(inpainted_image)
        
        # 5개 inpaint된 이미지에 대한 depth 예측값을 계산
        batch_image = torch.cat([transform(img).to(device).unsqueeze(0) for img in inpainted_images], dim=0)
        batch_depth_images = []
        
        for img in batch_image:
            depth_image = predict(model, img.unsqueeze(0), name, output_type, date, test, image_num)
            batch_depth_images.append(depth_image)

        # 5개의 예측 결과로 median 처리
        median_image = median_depth(batch_depth_images)
        
        # 유효 마스크 크기 맞추기
        valid_mask = cv2.resize(glass_mask, (640, 352))
        
        # 유효 마스크 영역에 대해 중간값을 최대값으로 설정
        valid_mask_nonzero = valid_mask > 0
        if np.any(valid_mask_nonzero):
            # 유효 마스크 영역에서만 min_value 계산
            min_value = np.min(median_image[valid_mask_nonzero])
            print(min_value)
            # 유효 마스크 영역에 대해 최소값을 적용
            median_image[valid_mask_nonzero] = min_value
        # else:
            # print(f"No valid mask found for {name}. Skipping min_value application.")

        # 예측된 median 이미지를 저장
        output_path = f"./glass_hints/{date}/{test}/image_0{image_num}"
        os.makedirs(output_path, exist_ok=True)
        
        # median 이미지를 저장
        np.save(f"{output_path}/{name}", median_image)

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

    encoder_checkpoint = torch.load(f"/ssd1/jm_data/depth/ssl/depth-hints/logs/jbnu_stereo_finetuned/models/weights_{args.epoch}/encoder.pth", map_location=device)
    decoder_checkpoint = torch.load(f"/ssd1/jm_data/depth/ssl/depth-hints/logs/jbnu_stereo_finetuned/models/weights_{args.epoch}/depth.pth", map_location=device)

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
            mask_path = os.path.join(drive_path, f"mask/image_0{image_num}/data/")
            
            if os.path.isdir(input_path) and os.path.isdir(mask_path):
                print(f"Processing: {drive_name}/image_0{image_num}")
                visualization(model, device=device, input_path=input_path, 
                              mask_path=mask_path, transform=transform, 
                              output_type=output_type, date=args.date, test=drive_name, image_num=image_num)
            else:
                print(f"Skipped: {drive_name}/image_0{image_num} (missing folders)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Depth Prediction")
    parser.add_argument("--num_layers", type=int, default=18, help="Number of layers in ResNet encoder")
    parser.add_argument("--pretrained", type=bool, default=True, help="Use pretrained weights")
    parser.add_argument("--img_height", type=int, default=352, help="Input image height")
    parser.add_argument("--img_width", type=int, default=640, help="Input image width")
    parser.add_argument("--min_depth", type=float, default=0.1, help="Minimum depth value")
    parser.add_argument("--max_depth", type=float, default=40.0, help="Maximum depth value")
    parser.add_argument("--epoch", type=int, default=0, help="Epoch to load weights from")
    parser.add_argument("--date", type=str, default="2025_03_23", help="Epoch to load weights from")

    args = parser.parse_args()
    main(args)
