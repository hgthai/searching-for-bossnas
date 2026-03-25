"""
BossNAS++ Validation Hook: Masked Population-Centric Search + Learnable Soft Margin
=====================================================================================
Nâng cấp pha tìm kiếm/đánh giá cho BossNAS++:

1. **Masked Population-Centric Search** (Bước 4):
   - L_val_MIM: Tính giống training (Student masked → Teacher non-masked tại vị trí mask)
   - L_val_CLS: Student P_CLS vs trung bình P_CLS Teacher trên *toàn bộ quần thể*
     (entire architecture population, không chỉ sampled sub-networks)

2. **Learnable Soft Margin** (Bước 5 - Novelty):
   - Thay vì loại bỏ cứng (hard constraint) kiến trúc vượt ngưỡng MAdds:
     penalty = λ * max(0, MAdds(a) - Constraint)
   - λ (lambda) là nn.Parameter, được tối ưu tự động (learnable)
   - L_val_total = L_val_MIM + L_val_CLS + penalty

Reference: BossNAS++ / BossNAS Family paper.
"""

import datetime
import os
import re

try:
    import apex
    HAS_APEX = True
except ImportError:
    HAS_APEX = False
    print('apex is not installed')

import mmcv
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
import yaml
from mmcv.runner import Hook, obj_from_dict
from openselfsup import datasets
from openselfsup.hooks.registry import HOOKS
from openselfsup.utils import optimizers, print_log
from timm.utils import distribute_bn
from torch.utils.data import Dataset

from bossnas.models.masking import block_wise_masking, get_patch_level_mask
from bossnas.models.siamese_supernets.siamese_supernets_hytra_pp import (
    compute_mim_loss, compute_cls_loss,
)


def is_dist_initialized():
    """Check if distributed training is initialized."""
    return dist.is_available() and dist.is_initialized()


# ═══════════════════════════════════════════════════════════
#  LEARNABLE SOFT MARGIN PENALTY
# ═══════════════════════════════════════════════════════════

class AdaptiveSoftMargin(nn.Module):
    """
    Learnable Soft Margin Penalty cho architecture search.

    Thay vì hard constraint (loại bỏ thẳng tay kiến trúc vượt ngưỡng MAdds),
    sử dụng penalty term mềm với hệ số λ tự học.

    Formula (aligned with `soft_margin.py`):
        R(a) = max(0, (MAdds(a) - Constraint) / Constraint) ^ beta
        score = base_loss + lambda * R(a)

    Trong đó:
    - λ (lambda_param): nn.Parameter, được tối ưu qua gradient
      → Mô hình tự học mức độ phạt phù hợp
    - MAdds(a): MAdds (Multiply-Accumulate operations) của kiến trúc a
    - Constraint: Ngưỡng MAdds cho phép (budget)

    Ưu điểm so với Hard Constraint:
    - Smooth gradient: không có discontinuity tại boundary
    - Tự động cân bằng: λ tự điều chỉnh trade-off giữa accuracy và efficiency
    - Khám phá mềm: kiến trúc hơi vượt ngưỡng vẫn có cơ hội được xét

    Args:
        madds_constraint (float): Ngưỡng MAdds cho phép (triệu MACs).
            Default: 6000.0 (~6G MACs cho HyTra-scale models).
        initial_lambda (float): Giá trị khởi tạo cho λ. Default: 0.001.
    """

    def __init__(
            self,
            madds_constraint=3.4 * 10**9,
            alpha=1e-9,
            beta=2.0,
            init_lambda=0.1,
            init_delta=0.0):
        super(AdaptiveSoftMargin, self).__init__()
        self.madds_constraint = float(madds_constraint)
        # Kept for backward compatibility in configs; not used in current formula.
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.lambda_param = nn.Parameter(torch.tensor(float(init_lambda), dtype=torch.float32))
        # Kept for backward compatibility in configs; not used in current formula.
        self.delta_param = nn.Parameter(torch.tensor(float(init_delta), dtype=torch.float32), requires_grad=False)

    def penalty(self, madds_value):
        if not isinstance(madds_value, torch.Tensor):
            madds_value = torch.tensor(madds_value, dtype=torch.float32,
                                       device=self.lambda_param.device)
        excess = (madds_value - self.madds_constraint) / self.madds_constraint
        return torch.clamp(excess, min=0.0) ** self.beta

    def forward(self, madds_value):
        """
        Tính penalty cho kiến trúc có MAdds = madds_value.

        Args:
            madds_value (float or torch.Tensor): MAdds của kiến trúc ứng viên (triệu MACs).

        Returns:
            penalty (torch.Tensor): Scalar penalty value.
                = λ * max(0, MAdds - Constraint)
                = 0 nếu kiến trúc nằm trong budget
                > 0 nếu kiến trúc vượt budget, tỷ lệ với mức vi phạm
        """
        raw_penalty = self.penalty(madds_value)
        weighted_penalty = torch.clamp_min(self.lambda_param, 0.0) * raw_penalty
        return weighted_penalty, raw_penalty

    def extra_repr(self):
        return (f'madds_constraint={self.madds_constraint}, '
            f'lambda={self.lambda_param.item():.6f}, '
            f'beta={self.beta}')


# ═══════════════════════════════════════════════════════════
#  MAdds ESTIMATION
# ═══════════════════════════════════════════════════════════

def _decode_hytra_op(op_idx):
    """Decode HyTra op id to (depth_bucket, op_kind)."""
    if op_idx < 2:
        return 0, ('ResAtt' if op_idx == 0 else 'ResConv')
    if op_idx < 4:
        return 1, ('ResAtt' if op_idx == 2 else 'ResConv')
    if op_idx == 4:
        return 2, 'ResAtt'
    if op_idx == 5:
        return 3, 'ResConv'
    return 0, ('ResAtt' if (op_idx % 2 == 0) else 'ResConv')


def _stage_block_madds(stage_idx, stage_path):
    """
    Estimate block MAdds from independent candidate layers in one stage.

    BossNAS++ uses modular block-wise search. We therefore accumulate
    per-layer cost inside each block and sum across blocks.
    """
    # Stage-dependent base cost per candidate layer (absolute MAdds).
    stage_layer_base = {
        0: [0.030e9, 0.045e9, 0.060e9, 0.070e9],
        1: [0.055e9, 0.075e9, 0.095e9, 0.115e9],
        2: [0.095e9, 0.130e9, 0.165e9, 0.205e9],
        3: [0.170e9, 0.230e9, 0.290e9, 0.370e9],
    }
    # Resolution-depth scaling. Deeper path -> smaller map -> lower MAdds.
    depth_scale = {
        0: 1.00,
        1: 0.84,
        2: 0.72,
        3: 0.62,
    }
    # Computation balancing in HyTra:
    # ResAtt uses lightweight implicit position encoding (depthwise separable),
    # so its cost is kept near ResConv for fair block comparison.
    op_scale = {
        'ResAtt': 1.02,
        'ResConv': 1.00,
    }

    base = stage_layer_base.get(stage_idx, stage_layer_base[3])
    total = 0.0
    for layer_idx, op_idx in enumerate(stage_path):
        if layer_idx >= len(base):
            break
        depth_bucket, op_kind = _decode_hytra_op(int(op_idx))
        total += base[layer_idx] * depth_scale.get(depth_bucket, 1.0) * op_scale.get(op_kind, 1.0)
    return total


def estimate_madds(model, path_encoding, input_size=(1, 3, 224, 224)):
    """
    Estimate total architecture MAdds (absolute count) in block-wise manner:

        total = stem_fixed + sum(selected_previous_blocks) + current_candidate_block
    """
    del input_size  # Kept for backward compatibility with old call signature.

    stem_fixed = 0.55e9
    total = stem_fixed

    best_paths = getattr(model, 'best_paths', [])
    for block_idx, block_path in enumerate(best_paths):
        total += _stage_block_madds(block_idx, block_path)

    start_block = int(getattr(model, 'start_block', 0))
    total += _stage_block_madds(start_block, path_encoding)
    return float(total)


# ═══════════════════════════════════════════════════════════
#  VALIDATION HOOK: Masked Population-Centric Search
# ═══════════════════════════════════════════════════════════

@HOOKS.register_module
class ValBestPathHookPP(Hook):
    """
    BossNAS++ Validation Hook: Masked Population-Centric Search
    với Learnable Soft Margin.

    Thay đổi so với ValBestPathHook gốc:
    ========================================
    1. L_val_MIM: Giống training → Student (masked) vs Teacher (non-masked)
    2. L_val_CLS: Student P_CLS vs trung bình P_CLS Teacher trên 
       *toàn bộ quần thể kiến trúc* (không chỉ sampled sub-nets)
    3. Soft Margin Penalty: Thay hard constraint bằng 
       penalty = λ * max(0, MAdds(a) - Constraint) với λ learnable

    Final evaluation loss:
        L_val_total = L_val_MIM + L_val_CLS + penalty

    Args:
        dataset: Validation dataset.
        bn_dataset: Dataset cho BatchNorm statistics update.
        interval: Evaluation interval (epochs).
        optimizer_cfg: Optimizer config cho stage transition.
        lr_cfg: Learning rate config.
        madds_constraint (float): MAdds budget (triệu MACs). Default: 6000.
        initial_lambda (float): Khởi tạo λ. Default: 0.001.
        masking_ratio (float): Tỷ lệ masking cho validation. Default: 0.3.
        patch_size (int): Patch size cho masking. Default: 16.
        dist_mode (bool): Distributed evaluation. Default: True.
    """

    def __init__(self,
                 dataset,
                 bn_dataset,
                 interval,
                 optimizer_cfg,
                 lr_cfg,
                 madds_constraint=3.4 * 10**9,
                 initial_lambda=0.1,
                 soft_margin_beta=2.0,
                 soft_margin_alpha=1e-9,
                 init_delta=0.0,
                 soft_margin_lr=1e-3,
                 num_generations=50,
                 topk_update=10,
                 topk_log=3,
                 masking_ratio=0.3,
                 patch_size=16,
                 dist_mode=True,
                 initial=True,
                 resume_best_path='',
                 supernet_checkpoint='',
                 epoch_per_stage=None,
                 **eval_kwargs):
        # ── Dataset setup ──
        if isinstance(dataset, Dataset) and isinstance(bn_dataset, Dataset):
            self.dataset = dataset
            self.bn_dataset = bn_dataset
        elif isinstance(dataset, dict) and isinstance(bn_dataset, dict):
            self.dataset = datasets.build_dataset(dataset)
            self.bn_dataset = datasets.build_dataset(bn_dataset)
        else:
            raise TypeError(
                f'dataset must be a Dataset object or a dict, not {type(dataset)}')

        self.data_loader = datasets.build_dataloader(
            self.dataset,
            eval_kwargs['imgs_per_gpu'],
            eval_kwargs['workers_per_gpu'],
            dist=dist_mode, shuffle=False,
            prefetch=eval_kwargs.get('prefetch', False),
            img_norm_cfg=eval_kwargs.get('img_norm_cfg', dict()))

        self.bn_data_loader = datasets.build_dataloader(
            self.bn_dataset,
            eval_kwargs['imgs_per_gpu'],
            eval_kwargs['workers_per_gpu'],
            dist=dist_mode, shuffle=True,
            prefetch=eval_kwargs.get('prefetch', False),
            img_norm_cfg=eval_kwargs.get('img_norm_cfg', dict()))
        self.bn_data = next(iter(self.bn_data_loader))
        self.bn_data = self.bn_data['img']
        del self.bn_data_loader

        self.dist_mode = dist_mode
        self.initial = initial
        self.interval = interval
        self.optimizer_cfg = optimizer_cfg
        self.lr_cfg = lr_cfg
        self.eval_kwargs = eval_kwargs
        self.epoch_per_stage = epoch_per_stage if epoch_per_stage is not None else interval
        self.masking_ratio = masking_ratio
        self.patch_size = patch_size
        self.madds_constraint = float(madds_constraint)
        self.topk_log = int(topk_log)
        self.num_generations = int(num_generations)
        self.topk_update = int(topk_update)
        self.supernet_checkpoint = supernet_checkpoint
        self._supernet_loaded = False

        self.adaptive_soft_margin = AdaptiveSoftMargin(
            madds_constraint=float(self.madds_constraint),
            alpha=float(soft_margin_alpha),
            beta=float(soft_margin_beta),
            init_lambda=float(initial_lambda),
            init_delta=float(init_delta),
        )
        self.soft_margin_optimizer = optim.Adam(
            [self.adaptive_soft_margin.lambda_param],
            lr=float(soft_margin_lr),
        )

        if resume_best_path:
            with open(resume_best_path, 'r') as f:
                self.loaded_best_path = yaml.load(f)
        else:
            self.loaded_best_path = []

    def after_train_epoch(self, runner):
        """
        Sau mỗi training epoch, thực hiện architecture evaluation.
        
        Flow:
        1. Đánh giá tất cả kiến trúc ứng viên
        2. Tính L_val_total = L_val_MIM + L_val_CLS + penalty
        3. Xếp hạng theo L_val_total  
        4. Chọn best path cho block hiện tại
        5. Transition sang block tiếp theo nếu đủ epochs
        """
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        block_interval = self.interval[model.start_block] \
            if isinstance(self.interval, list) else self.interval
        if not self.every_n_epochs(runner, block_interval):
            return

        if len(self.loaded_best_path) - 1 >= model.start_block:
            model.best_paths = self.loaded_best_path[:model.start_block + 1]
            print_log(f'loaded best paths: {model.best_paths}', logger='root')
        else:
            self._run_validate(runner)
            print_log(f'searched best paths (BossNAS++): {model.best_paths}', logger='root')

        print_log('best paths from all workers:')
        print(model.best_paths)
        torch.cuda.synchronize()

        # ── Save best path ──
        if runner.rank == 0:
            output_dir = os.path.join(runner.work_dir, 'path_rank')
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            time_str = datetime.datetime.strftime(datetime.datetime.now(), '%m-%d-%H')
            filename = os.path.join(
                output_dir,
                f"bestpath_pp_{model.start_block}_{runner.epoch}_{time_str}.yml"
            )
            with open(filename, 'w', encoding='utf8') as f:
                yaml.dump(model.best_paths, f)

            lambda_val = self.adaptive_soft_margin.lambda_param.item()
            print_log(
                f'AdaptiveSoftMargin params | lambda={lambda_val:.6f}, beta={self.adaptive_soft_margin.beta:.2f}',
                logger='root'
            )

        # ── Stage transition ──
        block_interval = self.epoch_per_stage[model.start_block] \
            if isinstance(self.epoch_per_stage, list) else self.epoch_per_stage
        if self.every_n_epochs(runner, block_interval) and \
                model.start_block < model.num_block - 1:
            model.start_block += 1
            forward_index = model.best_paths[-1][-1]
            if forward_index < 4:
                pos = forward_index // 2
            else:
                pos = forward_index - 2
            model.target_backbone.stage_depths[model.start_block] = pos + 1
            model.online_backbone.stage_depths[model.start_block] = pos + 1
            model.set_current_neck_and_head()
            del model.optimizer
            del runner.optimizer
            new_optimizer = build_optimizer(model, self.optimizer_cfg)
            if model.use_fp16 and HAS_APEX:
                model, new_optimizer = apex.amp.initialize(
                    model, new_optimizer, opt_level="O1")
            runner.optimizer = new_optimizer
            model.optimizer = new_optimizer

    def _run_validate(self, runner):
        """Chạy validation: Masked Population-Centric Search."""
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model

        if self.supernet_checkpoint and (not self._supernet_loaded):
            ckpt_path = self.supernet_checkpoint
            if not os.path.isabs(ckpt_path) and not os.path.exists(ckpt_path):
                fallback = os.path.join('src', 'searching', ckpt_path)
                if os.path.exists(fallback):
                    ckpt_path = fallback
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f'supernet_checkpoint not found: {self.supernet_checkpoint}')

            ckpt = torch.load(ckpt_path, map_location='cpu')
            state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
            model.load_state_dict(state_dict, strict=False)
            self._supernet_loaded = True

        for param in model.parameters():
            param.requires_grad = False

        runner.model.eval()
        results, metric_dict = self.multi_gpu_test_pp(runner, self.data_loader)

        # ── Sắp xếp theo L_val_total (ascending: loss thấp = tốt hơn) ──
        results = sorted(results.items(), key=lambda x: x[1], reverse=False)

        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        if runner.rank == 0:
            time_str = datetime.datetime.strftime(datetime.datetime.now(), '%m-%d-%H')
            output_dir = os.path.join(runner.work_dir, 'path_rank')
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            filename = os.path.join(
                output_dir,
                f"path_rank_pp_{model.start_block}_{time_str}.yml"
            )
            with open(filename, 'w', encoding='utf8') as f:
                yaml.dump(results, f)

            if results:
                topk = min(self.topk_log, len(results))
                print_log(
                    f'[Generation {runner.epoch}] Top-{topk} candidates (fitness thấp hơn là tốt hơn):',
                    logger='root'
                )
                for rank_idx, (op, fitness) in enumerate(results[:topk], start=1):
                    metrics = metric_dict.get(op, {})
                    acc_loss = metrics.get('accuracy_loss', float('nan'))
                    cost = metrics.get('cost', float('nan'))
                    penalty = metrics.get('penalty', float('nan'))
                    print_log(
                        f'  #{rank_idx} path={op} | fitness={fitness:.6f} | '
                        f'acc_loss={acc_loss:.6f} | cost={cost:.2f} | penalty={penalty:.6f}',
                        logger='root'
                    )

        block_interval = self.interval[model.start_block] \
            if isinstance(self.interval, list) else self.interval
        if self.every_n_epochs(runner, block_interval):
            best_path = results[0][0]
            best_path = [int(i) for i in list(best_path)]
            if len(model.best_paths) == model.start_block + 1:
                model.best_paths.pop()
            model.best_paths.append(best_path)

        runner.model.train()

    def multi_gpu_test_pp(self, runner, data_loader):
        """
        BossNAS++ Population-Centric Evaluation.

        Khác biệt chính so với BossNAS gốc:
        ========================================
        1. Áp dụng block-wise masking cho Student input
        2. Dùng BossNAS++ projectors (P_patch, P_CLS) 
        3. L_val_CLS target = trung bình P_CLS Teacher trên TOÀN BỘ quần thể
           (tất cả kiến trúc ứng viên, không chỉ sampled sub-nets)
        4. Thêm Soft Margin Penalty vào loss
        5. Cập nhật λ (learnable)

        Returns:
            tuple:
                - loss_dict: Dict {path_encoding_str: final_fitness}
                - metric_dict: Dict {path_encoding_str: {'accuracy_loss', 'cost', 'penalty'}}
        """
        if hasattr(runner.model, 'module'):
            model = runner.model.module
        else:
            model = runner.model
        model.eval()
        rank = runner.rank

        # ═══════════════════════════════════════════════════════
        #  PHASE 1: Thu thập Teacher P_CLS trên toàn bộ quần thể
        # ═══════════════════════════════════════════════════════
        # Đây là điểm khác biệt quan trọng nhất so với BossNAS gốc:
        # Target P_CLS được tính trung bình trên TẤT CẢ kiến trúc ứng viên
        # (entire population), không chỉ sampled sub-networks.

        all_path = model.online_backbone.get_all_path(start_block=model.start_block)

        # Accumulators cho mỗi data batch
        population_cls_v1_per_batch = []  # List[Tensor(B, D)]
        population_cls_v2_per_batch = []

        # Accumulators cho Student projections per path, per batch
        path_student_results = {}   # {op_str: {'patch_v1': [...], 'cls_v1': [...], ...}}
        path_online_results_v1 = {}  # BYOL projections cho backward compat
        path_online_results_v2 = {}
        mask_indices_per_batch_v1 = []  # Lưu mask cho mỗi batch
        mask_indices_per_batch_v2 = []
        feature_sizes_per_batch = []  # (H, W) cho mỗi batch

        # Target BYOL projections (averaged across population)
        avg_target_v1 = []
        avg_target_v2 = []

        if rank == 0:
            prog_bar = mmcv.ProgressBar(len(data_loader))

        for idx, data in enumerate(data_loader):
            with torch.no_grad():
                img = data['img']
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                img = img.to(device)
                assert img.dim() == 5, f"Input must have 5 dims, got: {img.dim()}"

                origin_img_v1 = img[:, 0, ...].contiguous()
                origin_img_v2 = img[:, 1, ...].contiguous()

                # ── Áp dụng block-wise masking cho Student ──
                masked_v1, mask_idx_v1 = block_wise_masking(
                    origin_img_v1, self.patch_size, self.masking_ratio
                )
                masked_v2, mask_idx_v2 = block_wise_masking(
                    origin_img_v2, self.patch_size, self.masking_ratio
                )
                mask_indices_per_batch_v1.append(mask_idx_v1)
                mask_indices_per_batch_v2.append(mask_idx_v2)

                # Accumulators per batch
                pop_cls_v1 = 0  # Teacher CLS sum across population
                pop_cls_v2 = 0
                temp_byol_v1 = 0
                temp_byol_v2 = 0

                for forward_singleop in all_path:
                    img_v1 = origin_img_v1  # Teacher uses NON-masked
                    img_v2 = origin_img_v2
                    img_v1_s = masked_v1  # Student uses masked
                    img_v2_s = masked_v2

                    op = ''.join([str(i) for i in forward_singleop])
                    if op not in path_student_results:
                        path_student_results[op] = {
                            'patch_v1': [], 'patch_v2': [],
                            'cls_v1': [], 'cls_v2': [],
                        }
                        path_online_results_v1[op] = []
                        path_online_results_v2[op] = []

                    # ── Update BN stats ──
                    update_bn_stats(self.bn_data, runner, forward_singleop)
                    model.eval()

                    # ── Forward qua resolved blocks ──
                    if model.start_block > 0:
                        for i, best_path in enumerate(model.best_paths):
                            img_v1 = model.target_backbone(
                                img_v1, start_block=i,
                                forward_op=best_path, block_op=True
                            )[0]
                            img_v2 = model.target_backbone(
                                img_v2, start_block=i,
                                forward_op=best_path, block_op=True
                            )[0]
                            img_v1_s = model.online_backbone(
                                img_v1_s, start_block=i,
                                forward_op=best_path, block_op=True
                            )[0]
                            img_v2_s = model.online_backbone(
                                img_v2_s, start_block=i,
                                forward_op=best_path, block_op=True
                            )[0]

                    # ── Teacher forward (non-masked) ──
                    feat_teacher_v1 = model.target_backbone(
                        img_v1, start_block=model.start_block,
                        forward_op=forward_singleop, block_op=True
                    )[0]
                    feat_teacher_v2 = model.target_backbone(
                        img_v2, start_block=model.start_block,
                        forward_op=forward_singleop, block_op=True
                    )[0]

                    # BYOL Target projections
                    proj_target_v1 = model.target_neck(
                        tuple([feat_teacher_v1])
                    )[0].clone().detach()
                    proj_target_v2 = model.target_neck(
                        tuple([feat_teacher_v2])
                    )[0].clone().detach()
                    temp_byol_v1 += proj_target_v1
                    temp_byol_v2 += proj_target_v2

                    # Teacher P_patch + P_CLS (population-centric)
                    t_patch_v1, t_cls_v1 = model.teacher_proj(feat_teacher_v1)
                    t_patch_v2, t_cls_v2 = model.teacher_proj(feat_teacher_v2)
                    pop_cls_v1 += t_cls_v1.detach()
                    pop_cls_v2 += t_cls_v2.detach()

                    # ── Student forward (masked) ──
                    feat_student_v1 = model.online_backbone(
                        img_v1_s, start_block=model.start_block,
                        forward_op=forward_singleop, block_op=True
                    )[0]
                    feat_student_v2 = model.online_backbone(
                        img_v2_s, start_block=model.start_block,
                        forward_op=forward_singleop, block_op=True
                    )[0]

                    # Student P_patch + P_CLS
                    s_patch_v1, s_cls_v1 = model.student_proj(feat_student_v1)
                    s_patch_v2, s_cls_v2 = model.student_proj(feat_student_v2)

                    # Teacher P_patch (per-path, for MIM loss)
                    path_student_results[op]['patch_v1'].append(
                        (s_patch_v1, t_patch_v1.detach()))
                    path_student_results[op]['patch_v2'].append(
                        (s_patch_v2, t_patch_v2.detach()))
                    path_student_results[op]['cls_v1'].append(s_cls_v1)
                    path_student_results[op]['cls_v2'].append(s_cls_v2)

                    # BYOL Student projections
                    proj_online_v1 = model.online_neck(
                        tuple([feat_student_v1])
                    )[0]
                    proj_online_v2 = model.online_neck(
                        tuple([feat_student_v2])
                    )[0]
                    path_online_results_v1[op].append(proj_online_v1)
                    path_online_results_v2[op].append(proj_online_v2)

                    if idx == 0:
                        feature_sizes_per_batch.append(
                            (feat_student_v1.shape[2], feat_student_v1.shape[3]))

                # ── Trung bình P_CLS Teacher trên toàn bộ quần thể ──
                population_cls_v1_per_batch.append(pop_cls_v1 / len(all_path))
                population_cls_v2_per_batch.append(pop_cls_v2 / len(all_path))
                avg_target_v1.append(temp_byol_v1 / len(all_path))
                avg_target_v2.append(temp_byol_v2 / len(all_path))

            if rank == 0:
                prog_bar.update()

        torch.cuda.synchronize()

        # ═══════════════════════════════════════════════════════
        #  PHASE 2: Tính L_base cho mỗi kiến trúc
        # ═══════════════════════════════════════════════════════
        base_loss_dict = {}
        cost_dict = {}

        for op, student_data in path_student_results.items():
            loss_val_total = 0
            num_batches = len(student_data['cls_v1'])

            for batch_idx in range(num_batches):
                # ── L_val_MIM: Tính giống training ──
                s_patch_v1, t_patch_v1 = student_data['patch_v1'][batch_idx]
                s_patch_v2, t_patch_v2 = student_data['patch_v2'][batch_idx]

                fh, fw = feature_sizes_per_batch[0] if feature_sizes_per_batch \
                    else (7, 7)

                loss_mim_v1 = compute_mim_loss(
                    s_patch_v1, t_patch_v1,
                    mask_indices_per_batch_v1[batch_idx], fh, fw
                )
                loss_mim_v2 = compute_mim_loss(
                    s_patch_v2, t_patch_v2,
                    mask_indices_per_batch_v2[batch_idx], fh, fw
                )
                loss_val_mim = (loss_mim_v1 + loss_mim_v2) / 2.0

                # ── L_val_CLS: Student P_CLS vs Population-averaged Teacher P_CLS ──
                # Điểm khác biệt: target là trung bình trên TOÀN BỘ quần thể
                s_cls_v1 = student_data['cls_v1'][batch_idx]
                s_cls_v2 = student_data['cls_v2'][batch_idx]
                pop_target_v1 = population_cls_v1_per_batch[batch_idx]
                pop_target_v2 = population_cls_v2_per_batch[batch_idx]

                loss_cls_v1 = compute_cls_loss(s_cls_v1, pop_target_v2)
                loss_cls_v2 = compute_cls_loss(s_cls_v2, pop_target_v1)
                loss_val_cls = (loss_cls_v1 + loss_cls_v2) / 2.0

                # ── BYOL Loss (backward compatible) ──
                proj_online_v1 = path_online_results_v1[op][batch_idx]
                proj_online_v2 = path_online_results_v2[op][batch_idx]
                loss_byol = model.head(proj_online_v1, avg_target_v2[batch_idx])['loss'] + \
                            model.head(proj_online_v2, avg_target_v1[batch_idx])['loss']

                # ── Tổng loss cho batch này ──
                batch_loss = loss_val_mim + loss_val_cls + loss_byol

                if is_dist_initialized():
                    loss_val_total += reduce_tensor(
                        batch_loss.data, dist.get_world_size()
                    ).item()
                else:
                    loss_val_total += batch_loss.item()

            loss_val_total /= max(num_batches, 1)

            path_encoding = [int(i) for i in list(op)]
            current_cost = estimate_madds(model, path_encoding)
            base_loss_dict[op] = float(loss_val_total)
            cost_dict[op] = float(current_cost)

        # ═══════════════════════════════════════════════════════
        #  PHASE 3-4: Hybrid Evolutionary-Gradient Search (50 generations)
        # ═══════════════════════════════════════════════════════
        device = self.adaptive_soft_margin.lambda_param.device
        final_loss_dict = {}
        final_metric_dict = {}

        for gen_idx in range(self.num_generations):
            tensor_total_dict = {}
            scalar_total_dict = {}
            metric_snapshot = {}

            for op, base_loss in base_loss_dict.items():
                base_tensor = torch.tensor(base_loss, dtype=torch.float32, device=device).detach()
                madds_tensor = torch.tensor(cost_dict[op], dtype=torch.float32, device=device)

                weighted_penalty, raw_penalty = self.adaptive_soft_margin(madds_tensor)
                total_loss = base_tensor + weighted_penalty

                tensor_total_dict[op] = total_loss
                scalar_total_dict[op] = float(total_loss.detach().item())
                metric_snapshot[op] = {
                    'accuracy_loss': float(base_loss),
                    'cost': float(cost_dict[op]),
                    'penalty': float(raw_penalty.detach().item()),
                    'weighted_penalty': float(weighted_penalty.detach().item()),
                }

            ranked = sorted(scalar_total_dict.items(), key=lambda x: x[1])
            if not ranked:
                break

            topk = min(self.topk_update, len(ranked))
            topk_ops = [op for op, _ in ranked[:topk]]
            mean_topk = torch.stack([tensor_total_dict[op] for op in topk_ops]).mean()

            self.soft_margin_optimizer.zero_grad(set_to_none=True)
            mean_topk.backward()

            self.soft_margin_optimizer.step()
            with torch.no_grad():
                self.adaptive_soft_margin.lambda_param.clamp_(min=0.0)

            if runner.rank == 0 and gen_idx in (0, self.num_generations - 1):
                print_log(
                    f'[HybridSearch gen={gen_idx + 1}/{self.num_generations}] '
                    f'mean_top{topk}={mean_topk.item():.6f} '
                    f'lambda={self.adaptive_soft_margin.lambda_param.item():.6f}',
                    logger='root'
                )

            if gen_idx == self.num_generations - 1:
                final_loss_dict = scalar_total_dict
                final_metric_dict = metric_snapshot

        return final_loss_dict, final_metric_dict


# ═══════════════════════════════════════════════════════════
#  UTILITIES (tái sử dụng từ val_hook.py gốc)
# ═══════════════════════════════════════════════════════════

def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt


def build_optimizer(model, optimizer_cfg):
    """Build optimizer (copy từ train.py để tránh circular import)."""
    if hasattr(model, 'module'):
        model = model.module
    optimizer_cfg = optimizer_cfg.copy()
    paramwise_options = optimizer_cfg.pop('paramwise_options', None)
    if paramwise_options is None:
        return obj_from_dict(optimizer_cfg, optimizers,
                             dict(params=model.parameters()))
    else:
        assert isinstance(paramwise_options, dict)
        params = []
        for name, param in model.named_parameters():
            param_group = {'params': [param]}
            if not param.requires_grad:
                params.append(param_group)
                continue
            for regexp, options in paramwise_options.items():
                if re.search(regexp, name):
                    for key, value in options.items():
                        if key.endswith('_mult'):
                            key = key[:-5]
                            assert key in optimizer_cfg
                            value = optimizer_cfg[key] * value
                        param_group[key] = value
            params.append(param_group)
        optimizer_cls = getattr(optimizers, optimizer_cfg.pop('type'))
        return optimizer_cls(params, **optimizer_cfg)


BN_MODULE_TYPES = (
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.SyncBatchNorm,
)


def update_bn_stats(bn_data, runner, forward_singleop):
    """Update BN statistics cho một path cụ thể."""
    bn_data = bn_data.cuda()
    assert bn_data.dim() == 5, f"Input must have 5 dims, got: {bn_data.dim()}"
    for layer in runner.model.modules():
        if isinstance(layer, BN_MODULE_TYPES):
            layer.reset_running_stats()
            layer.momentum = 1.
            layer.train()
    with torch.no_grad():
        runner.model(img=bn_data, mode='single', forward_singleop=forward_singleop)
    torch.cuda.synchronize()
    if is_dist_initialized():
        distribute_bn(runner.model, dist.get_world_size(), reduce=True)
