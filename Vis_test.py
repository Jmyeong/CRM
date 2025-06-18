import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm

# 입력 폴더와 출력 폴더 설정
base = False
epoch = 9

if base == True:
    print("Baseline visualization")
    split_folder = "jbnu_stereo_finetuned"
    
else:
    print("Ours visualization")
    split_folder = "jbnu_stereo_finetuned_glass"

input_folder = f"/ssd1/jm_data/depth/ssl/depth-hints/logs/{split_folder}/models/weights_{epoch}"  # .npy 파일이 있는 폴더 경로
output_folder = input_folder + f"/vis_base-{base}-{epoch}/"  # 변환된 .png 파일을 저장할 폴더

# 출력 폴더가 없으면 생성
os.makedirs(output_folder, exist_ok=True)

# 폴더 내 모든 npy 파일 처리
for filename in os.listdir(input_folder):
    if filename.endswith(".npy"):
        npy_path = os.path.join(input_folder, filename)

        # npy 파일 로드
        array = np.load(npy_path)
        # print(len(array))
        for i in tqdm(range(len(array))):
            tiff_path = os.path.join(output_folder, f"{i:010d}.tiff")
            # print(tiff_path)
            image = array[i]
            # print(image)
            # norm_image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)

            # 컬러맵 적용
            # colored_image = cm.magma(image)[:, :, :3]  # alpha 채널 제거 (RGB만 사용)

            # 0~255로 변환 후 저장
            # img_uint8 = (colored_image * 255).astype(np.uint8)
            print(np.unique(image))
            # print(img_uint8)
            # 이미지 저장
            Image.fromarray(image).save(tiff_path)
            # np.save(tiff_path, img_uint8)
            # print(tiff_path)
            # print(f"Saved: {tiff_path}")

print("변환 완료 ✅")
