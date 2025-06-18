import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 입력 이미지와 깊이 맵이 저장된 폴더 경로
dirname = "image_03"
image_folder = f'/ssd1/jm_data/depth/ssl/monodepth2/jbnu_stereo/2025_02_28/2025_02_28_drive_0003_sync/{dirname}/data'  # 이미지 폴더 경로
depth_map_folder = f'/ssd1/jm_data/depth/ssl/monodepth2/jbnu_stereo/2025_02_28/2025_02_28_drive_0003_sync/proj_depth/groundtruth/{dirname}'
output_folder = './outputs'  # 결과 이미지 저장 폴더 경로

# 결과 폴더가 없으면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 폴더 내의 모든 이미지 파일 이름 가져오기
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
gt_files = [f for f in os.listdir(depth_map_folder) if f.endswith(('.npy'))]
# 이미지 파일을 이름 순서대로 정렬
image_files.sort()
gt_files.sort()
# print(gt_files)
# 각 이미지에 대해 처리
for image_file, gt_file in zip(image_files, gt_files):
    image_path = os.path.join(image_folder, image_file)
    # print(depth_map_folder)
    # print(image_file)
    depth_map_path = os.path.join(depth_map_folder, gt_file)  # 동일한 이름의 깊이 맵
    if os.path.exists(depth_map_path):
        # 이미지와 깊이 맵 로드
        image = cv2.imread(image_path)
        # depth_map = cv2.imread(depth_map_path, cv2.IMREAD_UNCHANGED)
        depth_map = np.load(depth_map_path)
        # 깊이 맵을 정규화 후 색상 맵 적용
        depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_map_colored = cv2.applyColorMap(depth_map_normalized.astype(np.uint8), cv2.COLORMAP_JET)

        # 이미지와 깊이 맵 겹쳐서 오버레이
        alpha = 0.5  # 투명도 설정
        overlay = cv2.addWeighted(image, 1 - alpha, depth_map_colored, alpha, 0)

        # 결과 이미지 저장
        output_path = os.path.join(output_folder, f"overlay_{image_file}")
        cv2.imwrite(output_path, overlay)
        print(f"Saved overlay image: {output_path}")
    else:
        print(f"Depth map not found for {image_file}")
