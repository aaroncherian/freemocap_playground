from __future__ import annotations
from dataclasses import dataclass
import torch


def _gaussian_1d(mu: torch.Tensor, length: int, sigma: float) -> torch.Tensor:
    """
    mu: (N,) in bin coordinates
    returns (N, length) normalized distribution
    """
    x = torch.arange(length, device=mu.device).float()[None, :]  # (1, L)
    mu = mu.float()[:, None]                                     # (N, 1)
    g = torch.exp(-0.5 * ((x - mu) / float(sigma)) ** 2)
    g = g / (g.sum(dim=1, keepdim=True) + 1e-12)
    return g


@dataclass
class SimCCCodec:
    input_w: int
    input_h: int
    simcc_split_ratio: float = 2.0
    sigma_x: float = 6.0
    sigma_y: float = 6.93

    @property
    def Lx(self) -> int:
        return int(round(self.input_w * self.simcc_split_ratio))

    @property
    def Ly(self) -> int:
        return int(round(self.input_h * self.simcc_split_ratio))

    def encode(self, xy: torch.Tensor, valid: torch.Tensor):
        """
        xy: (B,K,2) in input pixel coords [0..W-1, 0..H-1]
        valid: (B,K) bool

        Returns:
          tx: (B,K,Lx)
          ty: (B,K,Ly)
        """
        B, K, _ = xy.shape
        device = xy.device

        Lx = self.Lx
        Ly = self.Ly

        # convert pixels -> bin coordinates
        x = xy[..., 0].clamp(0, self.input_w - 1) * self.simcc_split_ratio
        y = xy[..., 1].clamp(0, self.input_h - 1) * self.simcc_split_ratio

        v = valid.reshape(-1)
        x_flat = x.reshape(-1)[v]
        y_flat = y.reshape(-1)[v]

        tx = torch.zeros((B * K, Lx), device=device)
        ty = torch.zeros((B * K, Ly), device=device)

        if x_flat.numel() > 0:
            tx[v] = _gaussian_1d(x_flat, Lx, self.sigma_x)
            ty[v] = _gaussian_1d(y_flat, Ly, self.sigma_y)

        tx = tx.view(B, K, Lx)
        ty = ty.view(B, K, Ly)
        return tx, ty

    @torch.no_grad()
    def decode(self, logits_x: torch.Tensor, logits_y: torch.Tensor):
        """
        logits_x: (B,K,Lx), logits_y: (B,K,Ly)
        returns xy: (B,K,2) in input pixel coords
        """
        B, K, Lx = logits_x.shape
        Ly = logits_y.shape[-1]

        ix = logits_x.argmax(dim=-1).float()
        iy = logits_y.argmax(dim=-1).float()

        # map bins -> pixels
        x = ix / (Lx - 1) * (self.input_w * self.simcc_split_ratio - 1) / self.simcc_split_ratio
        y = iy / (Ly - 1) * (self.input_h * self.simcc_split_ratio - 1) / self.simcc_split_ratio
        return torch.stack([x, y], dim=-1)
