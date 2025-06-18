import os
import numpy as np
from PIL import Image

date = "2025_03_23"
drive = 13
image_num = 3

# 1. 경로 설정
base_path = f"/ssd1/jm_data/depth/ssl/monodepth2/jbnu_stereo/{date}/{date}_drive_00{drive}_sync"
image_folder = os.path.join(base_path, f"image_0{image_num}/data")
mask_folder = os.path.join(base_path, f"mask/image_0{image_num}/data")

# 2. 파일 목록 수집
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".png")])
mask_files = set([f for f in os.listdir(mask_folder) if f.endswith("_label.png")])

# 3. 마스크 사이즈 참조용 (임의 하나 로드)
sample_img_path = os.path.join(image_folder, image_files[0])
sample_img = Image.open(sample_img_path)
mask_size = sample_img.size[::-1]  # PIL은 (W, H), numpy는 (H, W)

# 4. 마스크가 없는 경우 0으로 채워서 저장
for img_file in image_files:
    index = os.path.splitext(img_file)[0]  # e.g. 0000000000
    mask_name = f"{index}_label.png"
    mask_path = os.path.join(mask_folder, mask_name)

    if mask_name not in mask_files:
        zero_mask = np.zeros(mask_size, dtype=np.uint8)  # H x W
        Image.fromarray(zero_mask).save(mask_path)
        print(f"✔️ 생성됨: {mask_path}")

print("✅ 누락된 마스크 파일 생성 완료.")
