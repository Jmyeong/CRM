import torch.nn as nn
import torch
import torch.nn.functional as F
import random
from layers import *

class FeatureFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureFusion, self).__init__()
        self.rgb_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 2, 1),
            nn.ELU(),
        )
        self.rgb_conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ELU(),
        )
        self.rgb_conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ELU(),
        )
        self.depth_conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 2, 1),
            nn.ELU(),
        )
        self.depth_conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ELU(),
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ELU()
        )
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ELU()
        )
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.ELU()
        )
        # self.conv1x1 = nn.Sequential(
        #     nn.Conv2d(1, 1, 3, 1, 1),
        #     nn.ELU()
        # )
        self.conv1x1 = Conv3x3(64, 64)
        self.dispconv = Conv3x3(1, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, rgb, depth):
        # print(rgb.shape, depth.shape)
        rgb1 = self.rgb_conv1(rgb)
        depth1 = self.depth_conv1(depth)
        rgb2 = self.rgb_conv2(rgb1 + depth1)
        depth2 = self.depth_conv2(depth1)
        encoded = self.final_conv(rgb2 + depth2)
        up1 = self.upsample1(encoded)
        # print(up1.shape, depth1.shape)
        out = self.conv1x1(up1)

        up2 = self.upsample2(out)
        # out = self.final_upsample(up2)

        out = self.dispconv(up2)
        out = self.sigmoid(out)
        return out, encoded
        
class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(1, 1, 3, 1, 1),
            nn.ReLU()
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x1):
        # x = torch.cat((x1, x2, x3, x4), dim=1) # B C*4 H W
        # print(x1.shape)
        return self.sigmoid(self.proj(x1))
        

class PatchExtractor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img, mask, patch_size):
        """
        img: (B, C, H, W) - input images
        mask: (B, 1, H, W) - binary mask (0 or 1)
        returns: (B, C, patch_size, patch_size)
        """
        B, C, H, W = img.shape
        device = img.device
        
        patches = []
        
        for b in range(B):
            mask_b = mask[b, 0]  # (H, W)

            ys, xs = torch.where(mask_b > 0)
            
            if len(ys) == 0:
                raise ValueError(f"Batch {b}: 마스크 안에 유효한 위치가 없습니다.")
            
            idx = random.randint(0, len(ys) - 1)
            y_center, x_center = ys[idx].item(), xs[idx].item()

            half_patch = patch_size // 2
            y1 = max(0, y_center - half_patch)
            y2 = min(H, y_center + half_patch)
            x1 = max(0, x_center - half_patch)
            x2 = min(W, x_center + half_patch)

            # 패치 잘라내기
            patch = img[b, :, y1:y2, x1:x2]  # (C, ?, ?)

            # 만약 패치가 patch_size보다 작으면 보간
            if patch.shape[1] != patch_size or patch.shape[2] != patch_size:
                patch = F.interpolate(patch.unsqueeze(0), size=(patch_size, patch_size), mode='bilinear', align_corners=False).squeeze(0)

            patches.append(patch)
        
        # (B, C, patch_size, patch_size)
        return torch.stack(patches, dim=0)