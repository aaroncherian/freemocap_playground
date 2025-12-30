from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def xy_scale_to_resized(xy: np.ndarray, W: int, H: int, out_w: int, out_h: int) -> np.ndarray:
    """Scale xy from original frame coords (W,H) -> resized coords (out_w,out_h)."""
    xy_rs = xy.copy()
    sx = out_w / max(1, W)
    sy = out_h / max(1, H)
    xy_rs[:, 0] *= sx
    xy_rs[:, 1] *= sy
    return xy_rs


class NPZVideoDatasetTopDown(Dataset):
    """
    Full-frame dataset:
      - reads frame fidx from video
      - resizes entire frame to (input_h,input_w)
      - scales GT points into resized space
      - returns img + xy (resized-space) + valid
    """
    def __init__(
        self,
        video_path: Path,
        labels_path: Path,
        frame_indices: list[int],
        input_w: int = 288,
        input_h: int = 384,
    ):
        self.video_path = Path(video_path)
        self.labels_path = Path(labels_path)
        self.frame_indices = frame_indices
        self.input_w = int(input_w)
        self.input_h = int(input_h)

        npz = np.load(self.labels_path, allow_pickle=True)
        self.points = npz["points"].astype(np.float32)        # (T,K,2) in original coords
        self.vis = npz["visibility"].astype(np.int32)         # (T,K)
        self.marker_names = [str(x) for x in npz["marker_names"]]
        self.T, self.K, _ = self.points.shape

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")
        self.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, i):
        fidx = int(self.frame_indices[i])
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError(f"Failed to read frame {fidx}")

        xy = self.points[fidx].copy()     # original coords
        vis = self.vis[fidx].copy()

        valid = (vis > 0) & np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])

        # Resize whole frame
        frame_rs = cv2.resize(frame, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)

        # Scale xy into resized coordinate system
        xy_rs = xy_scale_to_resized(xy, self.W, self.H, self.input_w, self.input_h)

        img = torch.from_numpy(frame_rs).permute(2, 0, 1).float() / 255.0
        img = (img - 0.5) / 0.5

        return {
            "img": img,
            "xy": torch.from_numpy(xy_rs).float(),          # (K,2) in resized coords
            "valid": torch.from_numpy(valid).bool(),        # (K,)
            "meta": {
                "frame_idx": fidx,
                "orig_wh": (self.W, self.H),
                "input_wh": (self.input_w, self.input_h),
            },
        }