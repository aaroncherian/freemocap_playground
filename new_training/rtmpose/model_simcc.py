from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class SimCCHead(nn.Module):
    def __init__(self, in_dim: int, num_keypoints: int, simcc_w: int, simcc_h: int):
        super().__init__()
        self.K = num_keypoints
        self.simcc_w = simcc_w
        self.simcc_h = simcc_h

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
        )
        self.out_x = nn.Linear(in_dim, self.K * self.simcc_w)
        self.out_y = nn.Linear(in_dim, self.K * self.simcc_h)

    def forward(self, feat: torch.Tensor):
        z = self.mlp(feat)
        lx = self.out_x(z).view(-1, self.K, self.simcc_w)
        ly = self.out_y(z).view(-1, self.K, self.simcc_h)
        return lx, ly


class RTMPoseTorch(nn.Module):
    def __init__(self, backbone: str, num_keypoints: int, simcc_w: int, simcc_h: int):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0, global_pool="avg")
        in_dim = self.backbone.num_features
        self.head = SimCCHead(in_dim, num_keypoints, simcc_w, simcc_h)

    def forward(self, img: torch.Tensor):
        feat = self.backbone(img)
        return self.head(feat)


def simcc_loss(lx, ly, tx, ty, valid):
    """
    lx: (B,K,simcc_w), ly: (B,K,simcc_h)
    tx,ty: (B,K)
    valid: (B,K) bool
    """
    B, K, _ = lx.shape
    v = valid.reshape(-1)
    if v.sum().item() == 0:
        return lx.sum() * 0.0

    lx2 = lx.reshape(B * K, -1)[v]
    ly2 = ly.reshape(B * K, -1)[v]
    tx2 = tx.reshape(-1)[v]
    ty2 = ty.reshape(-1)[v]

    return F.cross_entropy(lx2, tx2) + F.cross_entropy(ly2, ty2)


@torch.no_grad()
def simcc_decode(lx, ly, img_w: int, img_h: int):
    """
    Returns (B,K,2) in image coords of the *model input* space.
    """
    bx = lx.argmax(dim=-1).float()
    by = ly.argmax(dim=-1).float()
    simcc_w = lx.shape[-1]
    simcc_h = ly.shape[-1]

    x = bx / (simcc_w - 1) * (img_w - 1)
    y = by / (simcc_h - 1) * (img_h - 1)
    return torch.stack([x, y], dim=-1)
