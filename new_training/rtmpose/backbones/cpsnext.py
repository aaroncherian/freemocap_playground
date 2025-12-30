from __future__ import annotations
import torch
import torch.nn as nn
import timm


class CSPNeXtBackbone(nn.Module):
    """
    Thin wrapper around timm CSPNeXt models.
    Returns a pooled feature vector (B, C).
    """
    def __init__(self, model_name: str = "cspnext_small", pretrained: bool = True):
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained

        self.net = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        if not hasattr(self.net, "num_features"):
            raise RuntimeError(f"timm model {model_name} has no num_features; try a different model")

        self.out_dim = int(self.net.num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
