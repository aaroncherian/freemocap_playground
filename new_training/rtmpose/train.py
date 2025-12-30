from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from backbones import CSPNeXtBackbone
from heads import RTMCCHead
from new_training.rtmpose.codecs.simcc import SimCCCodec
from losses import KLDiscreteLoss
from data import NPZVideoDatasetTopDown


class RTMPoseLike(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_keypoints: int,
        input_w: int,
        input_h: int,
        simcc_split_ratio: float,
        pretrained_backbone: bool = True,
        head_hidden: int = 256,
    ):
        super().__init__()
        self.backbone = CSPNeXtBackbone(backbone_name, pretrained=pretrained_backbone)
        self.head = RTMCCHead(
            in_dim=self.backbone.out_dim,
            num_keypoints=num_keypoints,
            input_w=input_w,
            input_h=input_h,
            simcc_split_ratio=simcc_split_ratio,
            hidden_dim=head_hidden,
        )

    def forward(self, img):
        feat = self.backbone(img)
        return self.head(feat)


@dataclass
class TrainCfg:
    input_w: int = 512
    input_h: int = 512
    simcc_split_ratio: float = 2.0
    sigma_x: float = 6.0
    sigma_y: float = 6.93
    beta: float = 10.0

    pad: int = 80
    epochs: int = 60
    batch_size: int = 16
    lr: float = 5e-4
    seed: int = 0

    backbone_name: str = "cspnext_small"
    pretrained_backbone: bool = True


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(video_path: Path, labels_path: Path, ckpt_path: Path, cfg: TrainCfg):
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    npz = np.load(labels_path, allow_pickle=True)
    pts = npz["points"].astype(np.float32)
    vis = npz["visibility"].astype(np.int32)
    labeled = np.any(np.isfinite(pts[..., 0]) & np.isfinite(pts[..., 1]) & (vis > 0), axis=1)
    idxs = np.where(labeled)[0].tolist()
    random.shuffle(idxs)

    K = int(pts.shape[1])

    ds = NPZVideoDatasetTopDown(
        video_path=video_path,
        labels_path=labels_path,
        frame_indices=idxs,
        input_w=cfg.input_w,
        input_h=cfg.input_h,
    )
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

    model = RTMPoseLike(
        backbone_name=cfg.backbone_name,
        num_keypoints=K,
        input_w=cfg.input_w,
        input_h=cfg.input_h,
        simcc_split_ratio=cfg.simcc_split_ratio,
        pretrained_backbone=cfg.pretrained_backbone,
    ).to(device)

    codec = SimCCCodec(
        input_w=cfg.input_w,
        input_h=cfg.input_h,
        simcc_split_ratio=cfg.simcc_split_ratio,
        sigma_x=cfg.sigma_x,
        sigma_y=cfg.sigma_y,
    )

    loss_fn = KLDiscreteLoss(beta=cfg.beta, label_softmax=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    model.train()
    for ep in range(cfg.epochs):
        pbar = tqdm(dl, desc=f"epoch {ep+1}/{cfg.epochs}")
        for batch in pbar:
            img = batch["img"].to(device)
            xy = batch["xy"].to(device)          # (B,K,2)
            valid = batch["valid"].to(device)    # (B,K)
            if ep == 0 and pbar.n % 50 == 0:
                frac = valid.float().mean().item()
                any_per_sample = valid.any(dim=1).float().mean().item()
                print("valid frac:", frac, "samples with any kp:", any_per_sample)

            tx, ty = codec.encode(xy, valid)

            opt.zero_grad(set_to_none=True)
            lx, ly = model(img)

            loss = loss_fn(lx, tx, valid) + loss_fn(ly, ty, valid)
            loss.backward()
            opt.step()

            pbar.set_postfix(loss=float(loss.item()))

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    save_obj = {
        "model": model.state_dict(),
        "cfg": {
            **cfg.__dict__,
            "K": K,
        }
    }
    torch.save(save_obj, ckpt_path)
    print("Saved:", ckpt_path)


if __name__ == "__main__":
    path_to_data = Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1")

    video_path = path_to_data / "synchronized_videos" / "sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1_synced_Cam6.mp4"
    labels_path = path_to_data / "synchronized_videos" / "sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1_synced_Cam6.labels.npz"
    ckpt_path = path_to_data / "runs" / "rtmpose_like_cspnext.pt"

    cfg = TrainCfg(
        backbone_name="convnext_tiny",  # <-- set to a real timm model name on your machine
        pretrained_backbone=True,
    )
    main(video_path, labels_path, ckpt_path, cfg)
