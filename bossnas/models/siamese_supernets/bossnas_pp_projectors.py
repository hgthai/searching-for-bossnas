import torch
import torch.nn as nn


class BlockProjectors(nn.Module):
    """Project patch-level and cls-level features for BossNAS++."""

    def __init__(self, in_channels, hidden_dim=2048, out_dim=256):
        super(BlockProjectors, self).__init__()
        self.patch_projector = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_dim, kernel_size=1, bias=False),
        )
        self.cls_projector = nn.Sequential(
            nn.Linear(in_channels, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=True),
        )

    def forward(self, x):
        # x: (B, C, H, W)
        patch_feat = self.patch_projector(x)
        # (B, C', H, W) -> (B, N, C')
        patch_feat = patch_feat.flatten(2).transpose(1, 2).contiguous()

        cls_feat = x.mean(dim=(2, 3))
        cls_feat = self.cls_projector(cls_feat)

        return patch_feat, cls_feat


@torch.no_grad()
def ema_update_projectors(student_projectors, teacher_projectors, momentum):
    for student_proj, teacher_proj in zip(student_projectors, teacher_projectors):
        for p_s, p_t in zip(student_proj.parameters(), teacher_proj.parameters()):
            p_t.data = p_t.data * momentum + p_s.data * (1.0 - momentum)
