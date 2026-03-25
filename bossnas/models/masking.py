"""
BossNAS++ Block-wise Masking Module
====================================
Implements block-wise masking for Masked Image Modeling (MIM) in BossNAS++.

Ý tưởng: Chia feature map (hoặc ảnh) thành các patch (block) có kích thước cố định,
sau đó che (mask) ngẫu nhiên một tỷ lệ nhất định các patch.
- Ảnh đã mask → đầu vào cho Student (online) supernet
- Ảnh gốc (non-masked) → đầu vào cho Teacher (EMA) supernet

Reference: BossNAS++ / Masked Image Modeling integration.
"""

import torch
import torch.nn as nn
import math


def block_wise_masking(x, patch_size=16, masking_ratio=0.3):
    """
    Áp dụng block-wise masking lên tensor ảnh.

    Chia ảnh thành các patch có kích thước `patch_size x patch_size`,
    sau đó mask ngẫu nhiên `masking_ratio` phần trăm các patch bằng cách
    đặt giá trị pixel trong patch đó về 0.

    Args:
        x (torch.Tensor): Tensor ảnh đầu vào, shape (B, C, H, W).
        patch_size (int): Kích thước mỗi patch (block). Default: 16.
        masking_ratio (float): Tỷ lệ patch bị mask (0.0 đến 1.0). Default: 0.3.

    Returns:
        x_tilde (torch.Tensor): Ảnh đã bị che khuất, shape (B, C, H, W).
        mask_indices (torch.Tensor): Boolean tensor đánh dấu vị trí các patch bị mask,
            shape (B, num_patches_h, num_patches_w). True = patch bị mask.
    """
    B, C, H, W = x.shape

    # ── Tính số lượng patch theo chiều cao và chiều rộng ──
    num_patches_h = H // patch_size  # số patch theo chiều cao
    num_patches_w = W // patch_size  # số patch theo chiều rộng
    num_patches = num_patches_h * num_patches_w  # tổng số patch

    # ── Số patch cần mask ──
    num_masked = int(math.ceil(masking_ratio * num_patches))

    # ── Tạo mask ngẫu nhiên cho mỗi sample trong batch ──
    # Mỗi sample có bộ mask riêng để tăng tính đa dạng
    # mask_flat: (B, num_patches), True tại vị trí bị mask
    mask_flat = torch.zeros(B, num_patches, dtype=torch.bool, device=x.device)

    for i in range(B):
        # Chọn ngẫu nhiên num_masked vị trí patch để mask
        perm = torch.randperm(num_patches, device=x.device)[:num_masked]
        mask_flat[i, perm] = True

    # ── Reshape mask thành dạng spatial 2D ──
    # (B, num_patches) → (B, num_patches_h, num_patches_w)
    mask_indices = mask_flat.view(B, num_patches_h, num_patches_w)

    # ── Mở rộng mask lên kích thước pixel để áp dụng lên ảnh ──
    # (B, num_patches_h, num_patches_w) → (B, 1, H', W') 
    # với H' = num_patches_h * patch_size, W' = num_patches_w * patch_size
    mask_spatial = mask_indices.unsqueeze(1).float()  # (B, 1, nH, nW)
    # Dùng repeat_interleave để "phóng to" mỗi giá trị mask thành patch_size x patch_size
    mask_spatial = mask_spatial.repeat_interleave(patch_size, dim=2)  # (B, 1, H', nW)
    mask_spatial = mask_spatial.repeat_interleave(patch_size, dim=3)  # (B, 1, H', W')

    # Xử lý trường hợp H, W không chia hết cho patch_size
    # → Pad mask_spatial cho khớp với kích thước ảnh gốc
    if mask_spatial.shape[2] < H or mask_spatial.shape[3] < W:
        pad_h = H - mask_spatial.shape[2]
        pad_w = W - mask_spatial.shape[3]
        mask_spatial = nn.functional.pad(mask_spatial, (0, pad_w, 0, pad_h), value=0)

    # ── Áp dụng mask: đặt pixel tại patch bị mask về 0 ──
    # mask_spatial: 1.0 = patch bị mask, 0.0 = giữ nguyên
    # → Nhân ảnh với (1 - mask) để zero-out các patch bị mask
    x_tilde = x * (1.0 - mask_spatial)

    return x_tilde, mask_indices


def get_patch_level_mask(mask_indices, feature_h, feature_w):
    """
    Chuyển đổi mask_indices (ở mức patch) sang mask ở mức feature map.

    Khi feature map có spatial size khác với grid patch ban đầu,
    hàm này interpolate mask cho khớp với kích thước feature map.

    Args:
        mask_indices (torch.Tensor): Boolean mask, shape (B, nH_patch, nW_patch).
        feature_h (int): Chiều cao của feature map.
        feature_w (int): Chiều rộng của feature map.

    Returns:
        feature_mask (torch.Tensor): Boolean mask tại mức feature map,
            shape (B, feature_h, feature_w). True = vị trí bị mask.
    """
    B = mask_indices.shape[0]

    # ── Chuyển sang float để dùng interpolate ──
    mask_float = mask_indices.unsqueeze(1).float()  # (B, 1, nH, nW)

    # ── Interpolate (nearest) sang kích thước feature map ──
    # Dùng nearest để giữ nguyên binary mask (không bị blur)
    feature_mask = nn.functional.interpolate(
        mask_float,
        size=(feature_h, feature_w),
        mode='nearest'
    )  # (B, 1, feature_h, feature_w)

    # ── Chuyển lại về boolean ──
    feature_mask = feature_mask.squeeze(1).bool()  # (B, feature_h, feature_w)

    return feature_mask


class BlockWiseMasking(nn.Module):
    """
    Module wrapper cho block-wise masking.
    
    Có thể dùng như một nn.Module trong pipeline PyTorch.
    Hỗ trợ chế độ training (có mask) và eval (không mask).

    Args:
        patch_size (int): Kích thước mỗi patch. Default: 16.
        masking_ratio (float): Tỷ lệ mask. Default: 0.3.
    """

    def __init__(self, patch_size=16, masking_ratio=0.3):
        super(BlockWiseMasking, self).__init__()
        self.patch_size = patch_size
        self.masking_ratio = masking_ratio

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input image tensor, shape (B, C, H, W).

        Returns:
            x_tilde (torch.Tensor): Masked image, shape (B, C, H, W).
            mask_indices (torch.Tensor): Patch-level mask, shape (B, nH, nW).
        """
        if self.training:
            return block_wise_masking(x, self.patch_size, self.masking_ratio)
        else:
            # Ở chế độ eval, không mask → trả về ảnh gốc và mask toàn False
            B, C, H, W = x.shape
            nH = H // self.patch_size
            nW = W // self.patch_size
            mask_indices = torch.zeros(B, nH, nW, dtype=torch.bool, device=x.device)
            return x, mask_indices

    def extra_repr(self):
        return f'patch_size={self.patch_size}, masking_ratio={self.masking_ratio}'
