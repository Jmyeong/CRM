import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from collections import defaultdict

class CRM(nn.Module):
    def __init__(self, intrinsic: np.ndarray):
        super().__init__()
        self.K_inv = torch.from_numpy(np.linalg.inv(intrinsic)).float()

    @staticmethod
    def backproject_depth(depth: torch.Tensor, inv_K: torch.Tensor):
        if depth.ndim == 3:
            depth = depth.squeeze(0)
        H, W = depth.shape
        u = torch.arange(W, device=depth.device)
        v = torch.arange(H, device=depth.device)
        u, v = torch.meshgrid(u, v, indexing='xy')
        ones = torch.ones_like(u)
        pix_coords = torch.stack([u, v, ones], dim=0).reshape(3, -1).float()
        cam_dirs = torch.matmul(inv_K, pix_coords)
        depth_flat = depth.reshape(-1)
        cam_points = cam_dirs * depth_flat
        return cam_points.T.reshape(H, W, 3)

    @staticmethod
    def normalize_vector(v, eps=1e-6):
        return v / (torch.norm(v, dim=-1, keepdim=True) + eps)

    def get_surface_normal(self, cam_points: torch.Tensor, nei: int = 1):
        H, W, _ = cam_points.shape
        pad = nei
        ctr = cam_points[pad:-pad, pad:-pad]
        x0, x1 = cam_points[pad:-pad, 0:-(2 * pad)], cam_points[pad:-pad, 2 * pad:]
        y0, y1 = cam_points[0:-(2 * pad), pad:-pad], cam_points[2 * pad:, pad:-pad]
        x0y0, x0y1 = cam_points[0:-(2 * pad), 0:-(2 * pad)], cam_points[2 * pad:, 0:-(2 * pad)]
        x1y0, x1y1 = cam_points[0:-(2 * pad), 2 * pad:], cam_points[2 * pad:, 2 * pad:]

        n0 = self.normalize_vector(torch.cross(x0 - ctr, y0 - ctr, dim=-1))
        n1 = self.normalize_vector(torch.cross(x1 - ctr, y1 - ctr, dim=-1))
        n2 = self.normalize_vector(torch.cross(x0y0 - ctr, x0y1 - ctr, dim=-1))
        n3 = self.normalize_vector(torch.cross(x1y0 - ctr, x1y1 - ctr, dim=-1))

        normals = self.normalize_vector((n0 + n1 + n2 + n3) / 4.0)
        normal_map = torch.zeros_like(cam_points)
        normal_map[pad:-pad, pad:-pad] = normals
        return normal_map

    @staticmethod
    def get_connected_components(mask):
        if isinstance(mask, torch.Tensor):
            mask_np = mask.squeeze(0).detach().cpu().numpy().astype(np.uint8)
        else:
            mask_np = mask.astype(np.uint8)

        num_labels, labels = cv2.connectedComponents(mask_np)
        return [torch.from_numpy((labels == label).astype(np.uint8)).to(mask.device) for label in range(1, num_labels)]
    
    @staticmethod
    def get_mask_side_regions(mask: torch.Tensor, normals: torch.Tensor, side_width: int = 20):
        mask_f = mask.float().unsqueeze(0).unsqueeze(0)
        grad_x = F.pad(mask_f, (0, 1, 0, 0))[:, :, :, 1:] - mask_f
        grad_x = grad_x.squeeze()
        left_edge, right_edge = (grad_x == 1), (grad_x == -1)
        kernel = torch.ones((1, 1, 1, side_width * 2 + 1), dtype=torch.float32, device=mask.device)
        left_dilate = F.conv2d(left_edge.float().unsqueeze(0).unsqueeze(0), kernel, padding=(0, side_width)).squeeze() > 0
        right_dilate = F.conv2d(right_edge.float().unsqueeze(0).unsqueeze(0), kernel, padding=(0, side_width)).squeeze() > 0
        not_mask = ~mask.bool()
        return left_dilate & not_mask, right_dilate & not_mask

    @staticmethod
    def get_inlier_coords(normals, region_mask, threshold=0.2, num_samples=2000):
        coords = torch.nonzero(region_mask, as_tuple=False)
        if coords.shape[0] == 0:
            return coords
        normal_vals = normals[region_mask]
        mean_normal = F.normalize(normal_vals.mean(dim=0), dim=0)
        cosine_sim = F.cosine_similarity(normal_vals, mean_normal.unsqueeze(0), dim=1)
        coords_inlier = coords[cosine_sim > threshold]
        N = coords_inlier.shape[0]
        if N >= num_samples:
            return coords_inlier[torch.linspace(0, N - 1, steps=num_samples).long()]
        elif N > 0:
            return torch.cat([coords_inlier, coords_inlier[-1:].repeat(num_samples - N, 1)], dim=0)
        else:
            return coords_inlier

    @staticmethod
    def interpolate_depth(depth, mask, coords_l, coords_r):
        """
        depth: (H, W)
        mask: (H, W)
        coords_l, coords_r: (N, 2) each
        """
        if depth.ndim > 2:
            depth = depth.squeeze(0)

        H, W = depth.shape
        device = depth.device
        new_depth = depth.clone()

        # y 범위만큼 배열 만들기
        y_indices = torch.arange(H, device=device)

        # (1) 좌우 좌표를 y 기준으로 모으기
        left_y = coords_l[:, 0]
        left_x = coords_l[:, 1]
        right_y = coords_r[:, 0]
        right_x = coords_r[:, 1]

        left_x_means = torch.zeros(H, device=device)  # y별 평균 x
        right_x_means = torch.zeros(H, device=device)
        valid_rows = torch.zeros(H, dtype=torch.bool, device=device)

        for y in y_indices:
            lx = left_x[left_y == y]
            rx = right_x[right_y == y]

            if lx.numel() > 0 and rx.numel() > 0:
                left_x_means[y] = lx.float().mean()
                right_x_means[y] = rx.float().mean()
                valid_rows[y] = True

        # (2) mask에서 유리문 영역만 찾아서 보간
        for y in y_indices[valid_rows]:
            row_mask = (mask[y] == 1)
            if row_mask.sum() == 0:
                continue
            x_coords = torch.where(row_mask)[0]

            lx = int(round(left_x_means[y].item()))
            rx = int(round(right_x_means[y].item()))

            ld = depth[y, lx]
            rd = depth[y, rx]

            if torch.isnan(ld) or torch.isnan(rd):
                continue

            ratios = (x_coords - lx) / (rx - lx + 1e-6)
            ratios = torch.clamp(ratios, 0.0, 1.0)

            interpolated = (1.0 - ratios) * ld + ratios * rd
            new_depth[y, x_coords] = interpolated

        return new_depth

    def forward(self, depth: torch.Tensor, mask: torch.Tensor):
        device = depth.device
        K_inv = self.K_inv.to(device)
        cam_points = self.backproject_depth(depth, K_inv)
        normals = self.get_surface_normal(cam_points)
        cluster_masks = self.get_connected_components(mask)
        # print(torch.unique(mask))
        # print(cluster_masks)
        refined_depth = depth.clone()

        for cluster_mask in cluster_masks:
            left_region, right_region = self.get_mask_side_regions(cluster_mask, normals)
            coords_l = self.get_inlier_coords(normals, left_region)
            coords_r = self.get_inlier_coords(normals, right_region)
            if coords_l.shape[0] == 0 or coords_r.shape[0] == 0:
                continue
            refined_depth = self.interpolate_depth(refined_depth, cluster_mask.bool(), coords_l, coords_r)

        return refined_depth

import torch
import torch.nn.functional as F


def backproject_depth_torch(depth, inv_K):
    """
    depth: (1, H, W)
    inv_K: (3, 3)
    return: (1, 3, H, W)
    """
    _, H, W = depth.shape
    device = depth.device

    u = torch.arange(0, W, device=device)
    v = torch.arange(0, H, device=device)
    grid_u, grid_v = torch.meshgrid(u, v, indexing='xy')
    ones = torch.ones_like(grid_u)
    pix_coords = torch.stack([grid_u, grid_v, ones], dim=0).float()  # (3, H, W)
    pix_coords = pix_coords.view(3, -1).unsqueeze(0)  # (1, 3, H*W)

    inv_K = inv_K.unsqueeze(0)  # (1, 3, 3)
    cam_dirs = torch.bmm(inv_K, pix_coords)  # (1, 3, H*W)
    depth_flat = depth.view(1, 1, -1)
    cam_points = cam_dirs * depth_flat  # (1, 3, H*W)
    return cam_points.view(1, 3, H, W)


def get_stereo_M_t2s_torch(baseline=0.12, device='cpu'):
    M = torch.eye(4, device=device).unsqueeze(0)  # (1, 4, 4)
    M[:, 0, 3] = -baseline
    return M


def reproject_to_source_torch(cam_points, K, M_t2s):
    """
    cam_points: (1, 3, H, W)
    K: (3, 3)
    M_t2s: (1, 4, 4)
    """
    _, _, H, W = cam_points.shape
    device = cam_points.device

    cam_points_flat = cam_points.view(1, 3, -1)  # (1, 3, H*W)
    ones = torch.ones((1, 1, H * W), device=device)
    cam_points_homo = torch.cat([cam_points_flat, ones], dim=1)  # (1, 4, H*W)

    cam_points_src = torch.bmm(M_t2s, cam_points_homo)[:, :3, :]  # (1, 3, H*W)
    K = K.unsqueeze(0)  # (1, 3, 3)
    pixels = torch.bmm(K, cam_points_src)  # (1, 3, H*W)
    p0 = pixels[:, :2, :] / (pixels[:, 2:3, :] + 1e-8)  # (1, 2, H*W)
    p0 = p0.view(1, 2, H, W).permute(0, 2, 3, 1)  # (1, H, W, 2)
    return p0


def warp_depth_to_source_view(depth, K, inv_K, M_t2s):
    cam_points = backproject_depth_torch(depth, inv_K)
    p0 = reproject_to_source_torch(cam_points, K, M_t2s)

    _, H, W = depth.shape
    p0_norm = p0.clone()
    p0_norm[..., 0] = (p0[..., 0] / (W - 1)) * 2 - 1
    p0_norm[..., 1] = (p0[..., 1] / (H - 1)) * 2 - 1

    warped = F.grid_sample(depth.unsqueeze(0), p0_norm, mode='bilinear', align_corners=True)
    return warped.squeeze(0), cam_points, p0  # (1, H, W)


def check_consistency(depth_tensor, K_tensor, baseline=0.12, z_thresh=0.02, visualize=False):
    """
    depth_tensor: (1, H, W) torch.Tensor
    K_tensor: (3, 3) torch.Tensor
    Returns: consistency_mask (1, H, W) torch.Tensor (1: valid, 0: invalid)
    """
    assert depth_tensor.ndim == 3 and depth_tensor.shape[0] == 1, "depth must be (1, H, W)"
    device = depth_tensor.device
    _, H, W = depth_tensor.shape

    K_tensor = K_tensor.to(device)
    inv_K_tensor = torch.inverse(K_tensor).to(device)

    # Stereo transformation matrices
    M_t2s = get_stereo_M_t2s_torch(baseline=baseline, device=device)
    M_s2t = torch.inverse(M_t2s)

    # Warp depth from target to source view
    depth_warped_to_src, cam_points, _ = warp_depth_to_source_view(depth_tensor, K_tensor, inv_K_tensor, M_t2s)

    # Reproject back to target frame
    cam_points_src = backproject_depth_torch(depth_warped_to_src, inv_K_tensor)
    cam_points_src_flat = cam_points_src.view(1, 3, -1)
    ones = torch.ones((1, 1, H * W), device=device)
    cam_points_src_homo = torch.cat([cam_points_src_flat, ones], dim=1)
    cam_points_src_to_target = torch.bmm(M_s2t, cam_points_src_homo)[:, :3, :].view(1, 3, H, W)

    # Z-difference
    z_diff = torch.abs(cam_points[0, 2] - cam_points_src_to_target[0, 2])
    consistency_mask = (z_diff < z_thresh).float().unsqueeze(0)  # (1, H, W)
    # print(torch.unique(consistency_mask))
    # Optional visualization
    if visualize:
        import matplotlib.pyplot as plt
        plt.imshow(consistency_mask.squeeze().cpu(), cmap='gray')
        plt.title("Consistency Mask")
        plt.axis("off")
        plt.show()
    # print("z_diff max:", z_diff.max().item())
    # print("z_diff mean:", z_diff.mean().item())
# check_consistency 내부
    # print("depth max:", depth_tensor.max().item())  # 0.03 이하면 정규화된 것


    return consistency_mask
