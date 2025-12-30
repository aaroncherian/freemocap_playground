from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class KLDiscreteLoss(nn.Module):
    """
    DLC-style KL discrete loss for SimCC distributions.
    """
    def __init__(self, beta: float = 10.0, label_softmax: bool = True):
        super().__init__()
        self.beta = float(beta)
        self.label_softmax = bool(label_softmax)

    def forward(self, logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """
        logits: (B,K,L)
        target: (B,K,L) (should sum to 1 for valid points)
        valid: (B,K) bool
        """
        B, K, L = logits.shape
        v = valid.reshape(-1)
        if v.sum().item() == 0:
            return logits.sum() * 0.0

        logits2 = logits.reshape(B * K, L)[v]
        target2 = target.reshape(B * K, L)[v]

        if self.label_softmax:
            target2 = F.softmax(target2 * self.beta, dim=-1)

        logp = F.log_softmax(logits2, dim=-1)
        return F.kl_div(logp, target2, reduction="batchmean")
