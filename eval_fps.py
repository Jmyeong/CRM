from __future__ import absolute_import, division, print_function

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import cv2
cv2.setNumThreads(0)

from utils import readlines
from options import MonodepthOptions
import datasets
import networks

# -----------------------------
# 설정
# -----------------------------
splits_dir = os.path.join(os.path.dirname(__file__), "splits")
mode = "total"        # "total" 또는 "transp" (transp_files.txt 사용)
N_IMAGES = 100        # 측정할 이미지 수
WARMUP_STEPS = 10     # 워밍업(모델/커널 안정화)

# -----------------------------
# 유틸
# -----------------------------
def percentile(arr, q):
    if len(arr) == 0:
        return float('nan')
    return float(np.percentile(arr, q))

def bytes_to_mb(x):
    return float(x) / (1024.0 ** 2)

# -----------------------------
# 모델 로드 (네 평가 코드 분기와 동일)
# -----------------------------
def load_models(opt):
    assert torch.cuda.is_available(), "CUDA is required."
    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
    assert os.path.isdir(opt.load_weights_folder), f"Cannot find weights folder: {opt.load_weights_folder}"
    print(f"-> Loading weights from {opt.load_weights_folder}")

    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

    enc_ckpt = torch.load(encoder_path, map_location="cpu")

    # encoder 선택 (평가 코드와 동일)
    if opt.backbone in ["resnet", "resnet_lite"]:
        encoder = networks.ResnetEncoderDecoder(
            num_layers=opt.num_layers, num_features=opt.num_features, model_dim=opt.model_dim
        )
    elif opt.backbone == "resnet18_lite":
        encoder = networks.LiteResnetEncoderDecoder(model_dim=opt.model_dim)
    elif opt.backbone == "eff_b5":
        encoder = networks.BaseEncoder.build(num_features=opt.num_features, model_dim=opt.model_dim)
    else:
        encoder = networks.Unet(
            pretrained=(not getattr(opt, "load_pretrained_model", False)),
            backbone=opt.backbone, in_channels=3, num_classes=opt.model_dim,
            decoder_channels=opt.dec_channels
        )

    # 디코더 선택
    if str(opt.backbone).endswith("_lite"):
        depth_decoder = networks.Lite_Depth_Decoder_QueryTr(
            in_channels=opt.model_dim, patch_size=opt.patch_size, dim_out=opt.dim_out,
            embedding_dim=opt.model_dim, query_nums=opt.query_nums, num_heads=4,
            min_val=opt.min_depth, max_val=opt.max_depth
        )
    else:
        depth_decoder = networks.Depth_Decoder_QueryTr(
            in_channels=opt.model_dim, patch_size=opt.patch_size, dim_out=opt.dim_out,
            embedding_dim=opt.model_dim, query_nums=opt.query_nums, num_heads=4,
            min_val=opt.min_depth, max_val=opt.max_depth
        )

    # 가중치 로드 (strict=False로 유연하게)
    enc_state = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in enc_ckpt.items() if k in enc_state}, strict=False)
    depth_decoder.load_state_dict(torch.load(decoder_path, map_location="cpu"))

    encoder.cuda().eval()
    depth_decoder.cuda().eval()
    torch.backends.cudnn.benchmark = True  # 고정 입력 크기에서 속도 ↑

    in_h = enc_ckpt.get('height', None)
    in_w = enc_ckpt.get('width', None)
    if in_h is not None and in_w is not None:
        print(f"-> Model input size: {in_w}x{in_h}")
    return encoder, depth_decoder, enc_ckpt

# -----------------------------
# 데이터 로더 (앞에서부터 배치=1)
# -----------------------------
def build_loader(opt, enc_ckpt):
    # split 이름: opt.split 우선, 없으면 opt.eval_split 사용
    split_name = getattr(opt, "split", None) or getattr(opt, "eval_split", None)
    assert split_name is not None, "opt.split 또는 opt.eval_split 중 하나가 필요합니다."

    if mode == "total":
        split_file = os.path.join(splits_dir, split_name, "test_files.txt")
    elif mode == "transp":
        split_file = os.path.join(splits_dir, split_name, "transp_files.txt")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    filenames = readlines(split_file)
    assert len(filenames) > 0, f"No files found in {split_file}"

    if getattr(opt, "eval_split", "") == "jbnu_stereo":
        dataset = datasets.JBNUDepthDataset(
            opt.data_path, filenames, enc_ckpt['height'], enc_ckpt['width'],
            [0], 1, is_train=False
        )
    else:
        dataset = datasets.KITTIRAWDataset(
            opt.data_path, filenames, enc_ckpt['height'], enc_ckpt['width'],
            [0], 1, is_train=False
        )

    return DataLoader(dataset, batch_size=1, shuffle=False,
                      num_workers=opt.num_workers, pin_memory=True, drop_last=False)

# -----------------------------
# 100장 평균 속도/메모리 측정 (추론만)
# -----------------------------
@torch.inference_mode()
def measure_avg_100(encoder, depth_decoder, loader):
    device = torch.device("cuda")

    # 워밍업
    it = iter(loader)
    for _ in range(WARMUP_STEPS):
        try:
            data = next(it)
        except StopIteration:
            break
        img = data[("color", 0, 0)].to(device, non_blocking=True)
        _ = depth_decoder(encoder(img))
    torch.cuda.synchronize()

    # 메모리 기준/피크 초기화
    baseline_alloc = torch.cuda.memory_allocated(device)
    baseline_res   = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)

    # 측정
    times_ms = []
    frames = 0
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev   = torch.cuda.Event(enable_timing=True)

    count = 0
    for data in loader:
        if count >= N_IMAGES:
            break
        img = data[("color", 0, 0)].to(device, non_blocking=True)

        start_ev.record()
        _ = depth_decoder(encoder(img))
        end_ev.record()
        torch.cuda.synchronize()

        times_ms.append(start_ev.elapsed_time(end_ev))  # ms
        frames += 1
        count += 1

    # 메모리 피크
    peak_alloc = torch.cuda.max_memory_allocated(device)
    peak_res   = torch.cuda.max_memory_reserved(device)
    inc_alloc  = max(0, peak_alloc - baseline_alloc)
    inc_res    = max(0, peak_res   - baseline_res)

    # 통계
    mean_ms = float(np.mean(times_ms)) if times_ms else float('nan')
    p50 = percentile(times_ms, 50); p90 = percentile(times_ms, 90); p95 = percentile(times_ms, 95)
    total_time_s = sum(times_ms) / 1000.0
    fps = (frames / total_time_s) if total_time_s > 0 else float('nan')

    stats = {
        "num_images": frames,
        "latency_ms": {"mean": mean_ms, "p50": p50, "p90": p90, "p95": p95},
        "fps_avg_over_images": fps,
        "gpu_memory_mb": {
            "baseline_alloc": bytes_to_mb(baseline_alloc),
            "peak_alloc":     bytes_to_mb(peak_alloc),
            "inc_alloc":      bytes_to_mb(inc_alloc),
            "baseline_reserved": bytes_to_mb(baseline_res),
            "peak_reserved":     bytes_to_mb(peak_res),
            "inc_reserved":      bytes_to_mb(inc_res),
        }
    }
    return stats

# -----------------------------
# txt 인자 파싱 지원(기존 방식)
# -----------------------------
def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)

def main():
    options = MonodepthOptions()
    # txt 파일 한 줄당 공백 구분 인자 파싱
    options.parser.convert_arg_line_to_args = convert_arg_line_to_args

    # 기존 스타일: python script.py args.txt  → 내부에서 '@args.txt'로 처리
    if len(sys.argv) == 2 and not sys.argv[1].startswith("@"):
        argfile = "@" + sys.argv[1]
        opt = options.parser.parse_args([argfile])
    else:
        opt = options.parse()

    encoder, depth_decoder, enc_ckpt = load_models(opt)
    loader = build_loader(opt, enc_ckpt)

    stats = measure_avg_100(encoder, depth_decoder, loader)

    lat = stats["latency_ms"]; mem = stats["gpu_memory_mb"]
    print(f"Images measured      : {stats['num_images']}")
    print(f"Avg latency per image: mean {lat['mean']:.2f} ms | p50 {lat['p50']:.2f} | p90 {lat['p90']:.2f} | p95 {lat['p95']:.2f}")
    print(f"Average FPS (100 imgs): {stats['fps_avg_over_images']:.2f}")
    print("GPU Memory (CUDA)    : "
          f"alloc baseline {mem['baseline_alloc']:.1f} MB -> peak {mem['peak_alloc']:.1f} MB "
          f"(+{mem['inc_alloc']:.1f} MB), "
          f"reserved baseline {mem['baseline_reserved']:.1f} MB -> peak {mem['peak_reserved']:.1f} MB "
          f"(+{mem['inc_reserved']:.1f} MB)")

if __name__ == "__main__":
    main()
