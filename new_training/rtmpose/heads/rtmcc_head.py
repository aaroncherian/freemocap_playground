from __future__ import annotations
import math
import torch
import torch.nn as nn


class RTMCCHead(nn.Module):
    """
    RTMCC / SimCC head:
    takes pooled features (B,C) and predicts
      logits_x: (B,K,Lx)
      logits_y: (B,K,Ly)

    Lx = input_w * split_ratio
    Ly = input_h * split_ratio
    """
    def __init__(
        self,
        in_dim: int,
        num_keypoints: int,
        input_w: int,
        input_h: int,
        simcc_split_ratio: float = 2.0,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.K = int(num_keypoints)
        self.input_w = int(input_w)
        self.input_h = int(input_h)
        self.split_ratio = float(simcc_split_ratio)

        self.Lx = int(round(self.input_w * self.split_ratio))
        self.Ly = int(round(self.input_h * self.split_ratio))

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(inplace=True),
        )

        self.fc_x = nn.Linear(hidden_dim, self.K * self.Lx)
        self.fc_y = nn.Linear(hidden_dim, self.K * self.Ly)

        # mild init helps stability
        nn.init.normal_(self.fc_x.weight, std=0.01)
        nn.init.normal_(self.fc_y.weight, std=0.01)
        nn.init.constant_(self.fc_x.bias, 0.0)
        nn.init.constant_(self.fc_y.bias, 0.0)

    def forward(self, feat: torch.Tensor):
        z = self.mlp(feat)
        lx = self.fc_x(z).view(-1, self.K, self.Lx)
        ly = self.fc_y(z).view(-1, self.K, self.Ly)
        return lx, ly
