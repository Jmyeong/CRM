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

drive = 8
image_num = 2
date = "2025_05_08"
test = f"{date}_drive_{str(drive).zfill(4)}_sync"
model_name = "jbnu_stereo_resnet_320x1024_student_do_crm"
# model_name = "jbnu_stereo_teacher_monodepth2"
# model_name = "kitti"
# test = "golf_cart_231117_case2"

def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction"""
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

def predict(model, image, name, output_type):
    predicts = {}
    predicts["depth"] = []

    features = model["encoder"](image)
    predicts["disparity"] = model["depth"](features)
    # print(predicts["disparity"])
    # KeyError 방지: 딕셔너리 키 접근 수정
    predicts["disparity"]["disp", 0] = F.interpolate(
        predicts["disparity"]["disp", 0], 
        size=(args.img_height, args.img_width), 
        mode="bilinear", 
        align_corners=False
    )
    # print(torch.unique(predicts["disparity"]["disp", 0]))
    # depth = predicts["disparity"]["disp", 0]
    _, depth = disp_to_depth(predicts["disparity"]["disp", 0], args.min_depth, args.max_depth)
    predicts["depth"] = depth  
    
    depth_numpy = depth.squeeze(0).squeeze(0).detach().cpu().numpy()
    print(np.unique(depth_numpy))
    output_path = f"./depth_vis/{output_type}/{test}/image_0{image_num}"
    os.makedirs(output_path, exist_ok=True)
    # print(np.unique(depth_numpy))
    # plt.imsave(f"{output_path}/{name}.png", depth_numpy, cmap="magma_r")
    np.save(f"{output_path}/{name}", depth_numpy)

def visualization(model, device, input_path, transform, output_type):
    files = sorted(glob(input_path + "*.png", recursive=False))
    print(input_path)
    # if len(file.split("/")) == 12:
    for file in tqdm(files):
        # print(file)
        name = os.path.basename(file).split(".")[0]
        image = Image.open(file).convert("RGB")
        image = transform(image).to(device).unsqueeze(0)
        predict(model, image=image, name=name, output_type=output_type)
    print("Depth maps saved")
    return files

def generate_video(rgbs, files_path, output_video):
    depth_maps = sorted(glob(os.path.join(files_path, "*.png")))
    print(len(depth_maps), len(rgbs))
    assert len(depth_maps) == len(rgbs)
    
    frame1 = cv2.imread(rgbs[0])
    frame2 = cv2.imread(depth_maps[0])
    height = min(frame1.shape[0], frame2.shape[0])
    width = min(frame1.shape[1], frame2.shape[1])
    
    video_writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), 15, (width * 2, height))
    
    for rgb_path, depth_path in zip(rgbs, depth_maps):
        rgb = cv2.imread(rgb_path)
        depth = cv2.imread(depth_path)
        # print(rgb.shape, depth.shape)
        if rgb.shape[:2] != depth.shape[:2]:  # 크기 조정 오류 방지
            depth = cv2.resize(depth, (width, height))
            rgb = cv2.resize(rgb, (width, height))

        # print(rgb.shape, depth.shape)
        combined_img = np.hstack((rgb, depth))
        # print(combined_img.shape)
        video_writer.write(combined_img)
    
    video_writer.release()
    print(f"Video saved: {output_video}")

def main(args):
    transform = transforms.Compose([
        transforms.Resize((args.img_height, args.img_width)),
        transforms.ToTensor(),
    ])
    
    output_type = "video"
    model = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model["encoder"] = networks.ResnetEncoderDecoder(args.num_layers, num_features=args.num_features, model_dim=args.model_dim).to(device)
    model["depth"] = networks.Lite_Depth_Decoder_QueryTr(in_channels=args.model_dim, patch_size=args.patch_size, dim_out=args.dim_out, embedding_dim=args.model_dim, 
                                                                    query_nums=args.query_nums, num_heads=4, min_val=args.min_depth, max_val=args.max_depth).to(device)

    encoder_checkpoint = torch.load(f"/ssd1/jm_data/depth/ssl/SfMNeXt-Impl/logs/{model_name}/models/weights_{args.epoch}/encoder.pth", map_location=device)
    decoder_checkpoint = torch.load(f"/ssd1/jm_data/depth/ssl/SfMNeXt-Impl/logs/{model_name}/models/weights_{args.epoch}/depth.pth", map_location=device)
    
    for key in ["height", "width", "use_stereo"]:
        encoder_checkpoint.pop(key, None)  # 불필요한 키 제거
    
    model["encoder"].load_state_dict(encoder_checkpoint)
    model["depth"].load_state_dict(decoder_checkpoint)
    
    # input_path = f"/ssd1/sz_data/sz_data/{test}/out_realsense_image/"
    input_path = f"/ssd1/jm_data/depth/ssl/monodepth2/jbnu_stereo/{date}/{date}_drive_{str(drive).zfill(4)}_sync/image_0{image_num}/data/"
    rgbs = visualization(model, device=device, input_path=input_path, transform=transform, output_type=output_type)
    
    if output_type == "video":
        generate_video(rgbs, files_path=f"./depth_vis/video/{test}/image_0{image_num}", output_video=f"./depth_vis/video/{test}/output.mp4")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Depth Prediction")
    parser.add_argument("--num_layers", type=int, default=50, help="Number of layers in ResNet encoder")
    parser.add_argument("--model_dim", type=int, default=32, help="Number of layers in ResNet encoder")
    parser.add_argument("--patch_size", type=int, default=20, help="Number of layers in ResNet encoder")
    parser.add_argument("--dim_out", type=int, default=128, help="Number of layers in ResNet encoder")
    parser.add_argument("--query_nums", type=int, default=128, help="Number of layers in ResNet encoder")
    parser.add_argument("--num_features", type=int, default=256, help="Number of layers in ResNet encoder")
    parser.add_argument("--pretrained", type=bool, default=True, help="Use pretrained weights")
    parser.add_argument("--img_height", type=int, default=352, help="Input image height")
    parser.add_argument("--img_width", type=int, default=640, help="Input image width")
    parser.add_argument("--min_depth", type=float, default=0.1, help="Minimum depth value")
    parser.add_argument("--max_depth", type=float, default=50.0, help="Maximum depth value")
    parser.add_argument("--epoch", type=int, default=0, help="Epoch to load weights from")
    
    args = parser.parse_args()
    main(args)