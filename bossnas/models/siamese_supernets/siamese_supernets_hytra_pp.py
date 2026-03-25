"""
BossNAS++ Siamese Supernet (SiameseSupernetsHyTraPP)
=====================================================
Phiên bản nâng cấp từ SiameseSupernetsHyTra, tích hợp:
1. Block-wise Masking (Masked Image Modeling)
2. Dual Projectors: P_patch (patch embeddings) + P_CLS (CLS token)
3. Masked Ensemble Bootstrapping (2 sub-networks sampling)
4. Loss: L_total = L_MIM + L_CLS per block
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

try:
    import apex
except:
    print('apex is not installed')

from openselfsup.models import builder
from openselfsup.models.registry import MODELS

from bossnas.models.masking import block_wise_masking, get_patch_level_mask
from bossnas.models.siamese_supernets.bossnas_pp_projectors import (
    BlockProjectors,
    ema_update_projectors,
)


def is_dist_initialized():
    """Check if distributed training is initialized."""
    return dist.is_available() and dist.is_initialized()


# ═══════════════════════════════════════════════════════════
#  LOSS FUNCTIONS cho BossNAS++
# ═══════════════════════════════════════════════════════════

def compute_mim_loss(student_patch_proj, teacher_patch_proj, mask_indices, feature_h, feature_w):
    """
    Tính L_MIM: Masked Image Modeling loss.
    
    Cross-Entropy loss giữa output P_patch (Student) tại các vị trí *bị mask*
    và P_patch (Teacher EMA) tại các vị trí tương ứng.

    Cụ thể: Student phải "dự đoán" representation của Teacher
    tại những patch mà Student không được thấy (bị mask).

    Args:
        student_patch_proj (torch.Tensor): Student patch projections, 
            shape (B, num_patches, D). D = output dim của P_patch.
        teacher_patch_proj (torch.Tensor): Teacher patch projections (detached),
            shape (B, num_patches, D).
        mask_indices (torch.Tensor): Boolean mask ở mức patch grid,
            shape (B, nH_patch, nW_patch). True = patch bị mask.
        feature_h (int): Chiều cao feature map.
        feature_w (int): Chiều rộng feature map.

    Returns:
        loss_mim (torch.Tensor): Scalar MIM loss.
    """
    B, N, D = student_patch_proj.shape

    # ── Chuyển mask_indices sang kích thước feature map ──
    feature_mask = get_patch_level_mask(mask_indices, feature_h, feature_w)  # (B, fH, fW)
    # Flatten spatial: (B, fH, fW) → (B, fH*fW)
    feature_mask_flat = feature_mask.view(B, -1)  # (B, N)

    # ── Đảm bảo số patches khớp nhau ──
    # Nếu N khác fH*fW (do interpolation), resize mask
    if feature_mask_flat.shape[1] != N:
        feature_mask_flat = F.interpolate(
            feature_mask.unsqueeze(1).float(),
            size=(int(N ** 0.5), int(N ** 0.5)),
            mode='nearest'
        ).squeeze(1).bool().view(B, -1)

    # ── Chỉ lấy embeddings tại các vị trí BỊ MASK ──
    # Đây là điểm khác biệt cốt lõi: Student phải dự đoán những gì nó không thấy
    masked_student = []
    masked_teacher = []
    for i in range(B):
        mask_i = feature_mask_flat[i]  # (N,)
        if mask_i.sum() > 0:
            masked_student.append(student_patch_proj[i, mask_i])  # (num_masked, D)
            masked_teacher.append(teacher_patch_proj[i, mask_i])  # (num_masked, D)

    if len(masked_student) == 0:
        # Edge case: không có patch nào bị mask
        return torch.tensor(0.0, device=student_patch_proj.device, requires_grad=True)

    # ── Concatenate tất cả masked patches từ batch ──
    masked_student = torch.cat(masked_student, dim=0)  # (total_masked, D)
    masked_teacher = torch.cat(masked_teacher, dim=0)  # (total_masked, D)

    # ── Normalize embeddings (chuẩn L2) ──
    masked_student = F.normalize(masked_student, dim=-1)
    masked_teacher = F.normalize(masked_teacher, dim=-1)

    # ── Tính Cross-Entropy Loss ──
    # Dùng Teacher projections làm "soft labels" (target distribution)
    # Student cần match distribution này
    # Cosine similarity → logits → cross-entropy
    temperature = 0.1  # temperature scaling cho softmax
    logits = torch.mm(masked_student, masked_teacher.T) / temperature  # (M, M)
    # Target: mỗi student patch phải match với chính teacher patch tương ứng (diagonal)
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_mim = F.cross_entropy(logits, labels)

    return loss_mim


def compute_cls_loss(student_cls_proj, teacher_cls_proj_ensemble):
    """
    Tính L_CLS: CLS token distillation loss.
    
    Cross-Entropy loss giữa P_CLS của Student (với ảnh masked) 
    và trung bình cộng (probability ensemble) các P_CLS của Teacher 
    (với ảnh non-masked) từ tất cả sub-networks được lấy mẫu.

    Args:
        student_cls_proj (torch.Tensor): Student CLS projection,
            shape (B, D). D = output dim của P_CLS.
        teacher_cls_proj_ensemble (torch.Tensor): Averaged Teacher CLS projections
            (probability ensemble), shape (B, D). Đây là trung bình cộng 
            P_CLS của Teacher qua tất cả sub-networks.

    Returns:
        loss_cls (torch.Tensor): Scalar CLS loss.
    """
    # ── Normalize embeddings ──
    student_cls = F.normalize(student_cls_proj, dim=-1)  # (B, D)
    teacher_cls = F.normalize(teacher_cls_proj_ensemble, dim=-1)  # (B, D)

    # ── Tính Cross-Entropy Loss ──
    # Teacher ensemble làm target distribution (soft label)
    temperature = 0.1
    logits = torch.mm(student_cls, teacher_cls.T) / temperature  # (B, B)
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_cls = F.cross_entropy(logits, labels)

    return loss_cls


def compute_block_loss(student_patch_proj, teacher_patch_proj,
                       student_cls_proj, teacher_cls_proj_ensemble,
                       mask_indices, feature_h, feature_w):
    """
    Tính tổng Loss huấn luyện cho block k:
        L_total = L_MIM + L_CLS

    Args:
        student_patch_proj: (B, num_patches, D) - Student patch projections.
        teacher_patch_proj: (B, num_patches, D) - Teacher patch projections.
        student_cls_proj: (B, D) - Student CLS projection.
        teacher_cls_proj_ensemble: (B, D) - Averaged Teacher CLS projections.
        mask_indices: (B, nH, nW) - Patch-level mask.
        feature_h: int - Feature map height.
        feature_w: int - Feature map width.

    Returns:
        loss_total: Scalar total loss = L_MIM + L_CLS.
        loss_dict: Dict chứa chi tiết các thành phần loss.
    """
    # ── L_MIM: Masked Image Modeling loss ──
    loss_mim = compute_mim_loss(
        student_patch_proj, teacher_patch_proj,
        mask_indices, feature_h, feature_w
    )

    # ── L_CLS: CLS token distillation loss ──
    loss_cls = compute_cls_loss(student_cls_proj, teacher_cls_proj_ensemble)

    # ── Tổng loss ──
    loss_total = loss_mim + loss_cls

    loss_dict = {
        'loss_mim': loss_mim,
        'loss_cls': loss_cls,
        'loss_total': loss_total,
    }

    return loss_total, loss_dict


# ═══════════════════════════════════════════════════════════
#  SIAMESE SUPERNET BOSSNAS++ (HyTra)
# ═══════════════════════════════════════════════════════════

@MODELS.register_module
class SiameseSupernetsHyTraPP(nn.Module):
    """
    BossNAS++ Siamese Supernet cho HyTra search space.

    Nâng cấp so với SiameseSupernetsHyTra:
    - Tích hợp Block-wise Masking: ảnh masked → Student, ảnh gốc → Teacher
    - Dual Projectors: P_patch (MIM) + P_CLS (self-distillation) per block
    - Sampling: 2 sub-networks per step (thay vì 4 như bản gốc)
    - Loss: L_total = L_MIM + L_CLS
    - Teacher = EMA of Student (requires_grad=False)

    Args:
        backbone (dict): Config cho backbone (SupernetHyTra).
        neck (dict): Config cho projection neck (SimCLR-style).
        head (dict): Config cho prediction head (BYOL-style).
        start_block (int): Block bắt đầu huấn luyện.
        num_block (int): Tổng số block.
        pretrained (str): Đường dẫn pretrained weights.
        base_momentum (float): EMA momentum cơ bản. Default: 0.996.
        masking_ratio (float): Tỷ lệ block-wise masking. Default: 0.3.
        patch_size (int): Kích thước patch cho masking. Default: 16.
        proj_hidden_dim (int): Chiều ẩn cho P_patch/P_CLS. Default: 2048.
        proj_out_dim (int): Chiều đầu ra cho P_patch/P_CLS. Default: 256.
        num_sampled_subnets (int): Số sub-networks lấy mẫu mỗi step. Default: 2.
        use_fp16 (bool): Sử dụng mixed precision. Default: False.
        update_interval (int): Gradient accumulation interval.
    """

    def __init__(self,
                 backbone,
                 start_block,
                 num_block,
                 neck=None,
                 head=None,
                 pretrained=None,
                 base_momentum=0.996,
                 masking_ratio=0.3,
                 patch_size=16,
                 proj_hidden_dim=2048,
                 proj_out_dim=256,
                 num_sampled_subnets=2,
                 use_fp16=False,
                 update_interval=None,
                 **kwargs):
        super(SiameseSupernetsHyTraPP, self).__init__()

        self.start_block = start_block
        self.num_block = num_block
        self.masking_ratio = masking_ratio
        self.patch_size = patch_size
        self.num_sampled_subnets = num_sampled_subnets

        # ══════════════════════════════════════
        #  1. Build Backbones (Student & Teacher)
        # ══════════════════════════════════════
        self.online_backbone = builder.build_backbone(backbone)
        self.target_backbone = builder.build_backbone(backbone)
        self.backbone = self.online_backbone  # alias

        # ══════════════════════════════════════
        #  2. Build Necks (giữ nguyên từ BossNAS gốc cho backward compatibility)
        # ══════════════════════════════════════
        self.online_necks = nn.ModuleList()
        self.target_necks = nn.ModuleList()
        self.heads = nn.ModuleList()
        neck_in_channel_list = [cfg[0] for cfg in self.online_backbone.block_cfgs]
        for in_channel in neck_in_channel_list:
            self.online_necks.append(builder.build_neck(neck))
            self.target_necks.append(builder.build_neck(neck))
            self.heads.append(builder.build_head(head))

        # ══════════════════════════════════════
        #  3. Build BossNAS++ Projectors: P_patch + P_CLS per block
        # ══════════════════════════════════════
        # Mỗi block có resolution/channels riêng → cần projectors riêng
        # Student projectors: trainable
        # Teacher projectors: frozen (EMA)
        self.student_projectors = nn.ModuleList()
        self.teacher_projectors = nn.ModuleList()
        for in_channel in neck_in_channel_list:
            # Student projector cho block này
            student_proj = BlockProjectors(in_channel, proj_hidden_dim, proj_out_dim)
            self.student_projectors.append(student_proj)

            # Teacher projector cho block này (copy từ Student)
            teacher_proj = BlockProjectors(in_channel, proj_hidden_dim, proj_out_dim)
            for p_s, p_t in zip(student_proj.parameters(), teacher_proj.parameters()):
                p_t.data.copy_(p_s.data)
            # Đóng băng Teacher projectors
            for p in teacher_proj.parameters():
                p.requires_grad = False
            self.teacher_projectors.append(teacher_proj)

        # ══════════════════════════════════════
        #  4. Freeze Teacher (backbone + neck + projectors)
        # ══════════════════════════════════════
        for param in self.target_backbone.parameters():
            param.requires_grad = False
        for target_neck in self.target_necks:
            for param in target_neck.parameters():
                param.requires_grad = False
        # Teacher projectors đã freeze ở trên

        # ══════════════════════════════════════
        #  5. Initialize weights
        # ══════════════════════════════════════
        self.init_weights(pretrained=pretrained)
        self.set_current_neck_and_head()

        # ══════════════════════════════════════
        #  6. Training state
        # ══════════════════════════════════════
        self.base_momentum = base_momentum
        self.momentum = base_momentum
        self.forward_op_online = None
        self.forward_op_target = None
        self.best_paths = []
        self.optimizer = None
        self.use_fp16 = use_fp16
        self.update_interval = update_interval

    # ─── Initialization ────────────────────────────────

    def init_weights(self, pretrained=None):
        """Khởi tạo weights cho model."""
        self.online_backbone.init_weights()
        for online_neck in self.online_necks:
            online_neck.init_weights(init_linear='kaiming')

        # Copy Online → Target (backbone + necks)
        for p_ol, p_tgt in zip(self.online_backbone.parameters(),
                                self.target_backbone.parameters()):
            p_tgt.data.copy_(p_ol.data)
        for p_ol, p_tgt in zip(self.online_necks.parameters(),
                                self.target_necks.parameters()):
            p_tgt.data.copy_(p_ol.data)

        # Init prediction heads
        for head in self.heads:
            head.init_weights()

    def set_current_neck_and_head(self):
        """Set active neck/head/projectors dựa trên start_block hiện tại."""
        self.online_neck = self.online_necks[self.start_block]
        self.target_neck = self.target_necks[self.start_block]
        self.head = self.heads[self.start_block]

        # Set active projectors cho block hiện tại
        self.student_proj = self.student_projectors[self.start_block]
        self.teacher_proj = self.teacher_projectors[self.start_block]

        self.online_net = nn.Sequential(self.online_backbone, self.online_neck)
        self.target_net = nn.Sequential(self.target_backbone, self.target_neck)

    # ─── EMA Update ────────────────────────────────

    @torch.no_grad()
    def _momentum_update(self):
        """
        Momentum update cho tất cả Teacher components:
        - Target backbone
        - Target neck
        - Teacher projectors (P_patch + P_CLS)
        """
        # Update backbone + neck
        for p_ol, p_tgt in zip(self.online_net.parameters(),
                                self.target_net.parameters()):
            p_tgt.data = p_tgt.data * self.momentum + p_ol.data * (1. - self.momentum)

        # Update projectors (BossNAS++ addition)
        ema_update_projectors(
            self.student_projectors, self.teacher_projectors, self.momentum
        )

    @torch.no_grad()
    def momentum_update(self):
        self._momentum_update()

    # ─── Batch Shuffle/Unshuffle (DDP) ─────────────

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """Batch shuffle cho BatchNorm. Hỗ trợ cả distributed và non-distributed."""
        if not is_dist_initialized():
            batch_size = x.shape[0]
            idx_shuffle = torch.randperm(batch_size).cuda()
            idx_unshuffle = torch.argsort(idx_shuffle)
            return x[idx_shuffle], idx_unshuffle

        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]
        num_gpus = batch_size_all // batch_size_this
        idx_shuffle = torch.randperm(batch_size_all).cuda()
        torch.distributed.broadcast(idx_shuffle, src=0)
        idx_unshuffle = torch.argsort(idx_shuffle)
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]
        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """Undo batch shuffle."""
        if not is_dist_initialized():
            return x[idx_unshuffle]

        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]
        num_gpus = batch_size_all // batch_size_this
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]
        return x_gather[idx_this]

    # ─── Forward: Training ─────────────────────────

    def forward_train(self, img, forward_singleop_online, idx=0, **kwargs):
        """
        Forward computation cho BossNAS++ training.

        Quy trình:
        1. Tách ảnh thành 2 views (v1, v2)
        2. Áp dụng block-wise masking → x_masked (Student input)
        3. Forward Student backbone + projectors với ảnh masked
        4. Tính L_MIM + L_CLS sử dụng cached Teacher projections

        Args:
            img (Tensor): Input shape (N, num_views, C, H, W).
            forward_singleop_online: Encoding cho sub-network path.
            idx (int): View index.

        Returns:
            loss (Tensor): L_total = L_MIM + L_CLS + BYOL loss (backward compatible).
        """
        assert img.dim() == 5, f"Input must have 5 dims, got: {img.dim()}"

        v2_idx = img.shape[1] // 2
        img_v1 = img[:, idx, ...].contiguous()
        img_v2 = img[:, v2_idx + idx, ...].contiguous()

        # ── Bước 1: Áp dụng Block-wise Masking cho Student input ──
        # Ảnh masked → Student, ảnh gốc → Teacher (đã tính trong forward_target)
        img_v1_masked, mask_indices_v1 = block_wise_masking(
            img_v1, patch_size=self.patch_size, masking_ratio=self.masking_ratio
        )
        img_v2_masked, mask_indices_v2 = block_wise_masking(
            img_v2, patch_size=self.patch_size, masking_ratio=self.masking_ratio
        )

        # ── Bước 2: Forward qua các block đã giải quyết trước đó ──
        if self.start_block > 0:
            for i, best_path in enumerate(self.best_paths):
                img_v1_masked = self.online_backbone(
                    img_v1_masked, start_block=i, forward_op=best_path, block_op=True
                )[0]
                img_v2_masked = self.online_backbone(
                    img_v2_masked, start_block=i, forward_op=best_path, block_op=True
                )[0]

        # ── Bước 3: Forward Student backbone + Neck cho block hiện tại ──
        feat_v1 = self.online_backbone(
            img_v1_masked, start_block=self.start_block,
            forward_op=forward_singleop_online
        )[0]  # (B, C, H', W')
        feat_v2 = self.online_backbone(
            img_v2_masked, start_block=self.start_block,
            forward_op=forward_singleop_online
        )[0]

        # Neck projection (giữ nguyên cho BYOL compatibility)
        proj_online_v1 = self.online_neck(tuple([feat_v1]))[0]
        proj_online_v2 = self.online_neck(tuple([feat_v2]))[0]

        # ── Bước 4: Student Projectors (BossNAS++ addition) ──
        # P_patch: patch-level projections
        # P_CLS: global CLS projections
        student_patch_v1, student_cls_v1 = self.student_proj(feat_v1)
        student_patch_v2, student_cls_v2 = self.student_proj(feat_v2)

        feature_h, feature_w = feat_v1.shape[2], feat_v1.shape[3]

        # ── Bước 5: Tính BossNAS++ Loss ──
        # L_MIM: Student patch (masked) vs Teacher patch (non-masked) tại vị trí bị mask
        loss_mim_v1 = compute_mim_loss(
            student_patch_v1, self.teacher_patch_proj_v1,
            mask_indices_v1, feature_h, feature_w
        )
        loss_mim_v2 = compute_mim_loss(
            student_patch_v2, self.teacher_patch_proj_v2,
            mask_indices_v2, feature_h, feature_w
        )

        # L_CLS: Student CLS (masked) vs Teacher CLS ensemble (non-masked)
        loss_cls_v1 = compute_cls_loss(student_cls_v1, self.teacher_cls_ensemble_v2)
        loss_cls_v2 = compute_cls_loss(student_cls_v2, self.teacher_cls_ensemble_v1)

        # ── Bước 6: BYOL loss (backward compatible) ──
        loss_byol = self.head(proj_online_v1, self.proj_target_v2)['loss'] + \
                     self.head(proj_online_v2, self.proj_target_v1)['loss']

        # ── Bước 7: Tổng loss ──
        # L_total = L_MIM + L_CLS + L_BYOL
        loss_mim = (loss_mim_v1 + loss_mim_v2) / 2.0
        loss_cls = (loss_cls_v1 + loss_cls_v2) / 2.0
        loss = loss_byol + loss_mim + loss_cls

        return loss

    def forward_test(self, img, **kwargs):
        pass

    def forward(self, img, mode='train', **kwargs):
        if mode == 'train':
            return self.forward_train(img, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        elif mode == 'extract':
            return self.backbone(img)
        elif mode == 'target':
            return self.forward_target(img, **kwargs)
        elif mode == 'single':
            return self.forward_single(img, **kwargs)
        else:
            raise Exception(f"No such mode: {mode}")

    # ─── Forward: Teacher (Target) ─────────────────

    @torch.no_grad()
    def forward_target(self, img, **kwargs):
        """
        Forward Teacher network với ảnh NON-MASKED.

        BossNAS++ thay đổi:
        - Teacher nhận ảnh gốc (không mask)
        - Tính cả neck projections (BYOL) VÀ P_patch/P_CLS projections (MIM)
        - Lưu trữ Teacher CLS ensemble (trung bình P_CLS qua tất cả sub-nets)

        Args:
            img (Tensor): Shape (N, num_views, C, H, W).
        """
        assert img.dim() == 5, f"Input must have 5 dims, got: {img.dim()}"

        # ── Batch shuffle cho BatchNorm ──
        img_v_l = []
        idx_unshuffle_v_l = []
        for view_idx in range(img.shape[1]):
            img_vi = img[:, view_idx, ...].contiguous()
            img_vi, idx_unshuffle_vi = self._batch_shuffle_ddp(img_vi)
            img_v_l.append(img_vi)
            idx_unshuffle_v_l.append(idx_unshuffle_vi)

        # ── Forward qua các block đã giải quyết ──
        if self.start_block > 0:
            for view_idx in range(img.shape[1]):
                for i, best_path in enumerate(self.best_paths):
                    img_v_l[view_idx] = self.target_backbone(
                        img_v_l[view_idx], start_block=i,
                        forward_op=best_path, block_op=True
                    )[0]

        # ── Forward Teacher qua tất cả sampled sub-networks ──
        self.forward_op_target = self.forward_op_online  # dùng cùng sub-net paths
        v2_idx = img.shape[1] // 2

        proj_target_v1 = 0  # BYOL neck projections (averaged)
        proj_target_v2 = 0
        teacher_cls_v1 = 0  # P_CLS projections (averaged)
        teacher_cls_v2 = 0
        teacher_patch_v1 = 0  # P_patch projections (averaged)
        teacher_patch_v2 = 0

        with torch.no_grad():
            for op_idx, forward_singleop_target in enumerate(self.forward_op_target):
                # ── Forward backbone ──
                feat_v1 = self.target_backbone(
                    img_v_l[op_idx], start_block=self.start_block,
                    forward_op=forward_singleop_target
                )[0]
                feat_v2 = self.target_backbone(
                    img_v_l[v2_idx + op_idx], start_block=self.start_block,
                    forward_op=forward_singleop_target
                )[0]

                # ── Neck projection (BYOL) ──
                temp_v1 = self.target_neck(tuple([feat_v1]))[0].clone().detach()
                temp_v2 = self.target_neck(tuple([feat_v2]))[0].clone().detach()
                temp_v1 = F.normalize(temp_v1, dim=1)
                temp_v1 = self._batch_unshuffle_ddp(temp_v1, idx_unshuffle_v_l[op_idx])
                temp_v2 = F.normalize(temp_v2, dim=1)
                temp_v2 = self._batch_unshuffle_ddp(temp_v2, idx_unshuffle_v_l[v2_idx + op_idx])
                proj_target_v1 += temp_v1
                proj_target_v2 += temp_v2

                # ── BossNAS++ Projector projections ──
                # Batch unshuffle features trước khi tính projections
                feat_v1_unshuffled = self._batch_unshuffle_ddp(feat_v1, idx_unshuffle_v_l[op_idx])
                feat_v2_unshuffled = self._batch_unshuffle_ddp(feat_v2, idx_unshuffle_v_l[v2_idx + op_idx])

                # P_patch + P_CLS projections từ Teacher
                t_patch_v1, t_cls_v1 = self.teacher_proj(feat_v1_unshuffled)
                t_patch_v2, t_cls_v2 = self.teacher_proj(feat_v2_unshuffled)

                teacher_patch_v1 += t_patch_v1.clone().detach()
                teacher_patch_v2 += t_patch_v2.clone().detach()
                teacher_cls_v1 += t_cls_v1.clone().detach()
                teacher_cls_v2 += t_cls_v2.clone().detach()

        num_subnets = len(self.forward_op_target)

        # ── BYOL target (averaged across sub-networks) ──
        self.proj_target_v1 = proj_target_v1 / num_subnets
        self.proj_target_v2 = proj_target_v2 / num_subnets

        # ── BossNAS++ targets (averaged across sub-networks = probability ensemble) ──
        self.teacher_patch_proj_v1 = teacher_patch_v1 / num_subnets
        self.teacher_patch_proj_v2 = teacher_patch_v2 / num_subnets
        self.teacher_cls_ensemble_v1 = teacher_cls_v1 / num_subnets
        self.teacher_cls_ensemble_v2 = teacher_cls_v2 / num_subnets

    # ─── Forward: Single path (BN stats update) ───

    def forward_single(self, img, forward_singleop, **kwargs):
        """Forward single path để update BN stats (dùng khi validation)."""
        img_v1 = img[:, 0, ...].contiguous()
        img_v2 = img[:, 1, ...].contiguous()

        if self.start_block > 0:
            for i, best_path in enumerate(self.best_paths):
                img_v1 = self.target_backbone(
                    img_v1, start_block=i, forward_op=best_path, block_op=True
                )[0]
                img_v2 = self.target_backbone(
                    img_v2, start_block=i, forward_op=best_path, block_op=True
                )[0]

        self.target_neck(self.target_backbone(
            img_v1, start_block=self.start_block,
            forward_op=forward_singleop, block_op=True
        ))
        self.target_neck(self.target_backbone(
            img_v2, start_block=self.start_block,
            forward_op=forward_singleop, block_op=True
        ))
        self.online_neck(self.online_backbone(
            img_v1, start_block=self.start_block,
            forward_op=forward_singleop, block_op=True
        ))
        self.online_neck(self.online_backbone(
            img_v2, start_block=self.start_block,
            forward_op=forward_singleop, block_op=True
        ))


# ═══════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def concat_all_gather(tensor):
    """all_gather operation (no gradient)."""
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output
