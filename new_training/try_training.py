from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as tvm


# -------------------------
# Heatmaps
# -------------------------
def make_gaussian_heatmaps(points_xy, visible, H, W, sigma=2.5):
    K = points_xy.shape[0]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    out = np.zeros((K, H, W), dtype=np.float32)
    for k in range(K):
        if not visible[k]:
            continue
        x, y = points_xy[k]
        out[k] = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    return out


def decode_argmax_subpixel(hm: torch.Tensor):
    hm_np = hm.detach().cpu().numpy()
    K, H, W = hm_np.shape
    pts = np.zeros((K, 2), np.float32)
    conf = hm_np.reshape(K, -1).max(axis=1).astype(np.float32)

    def quad_offset(a, b, c):
        denom = (a - 2*b + c)
        if abs(denom) < 1e-8:
            return 0.0
        return 0.5 * (a - c) / denom

    for k in range(K):
        h = hm_np[k]
        idx = h.argmax()
        y, x = divmod(idx, W)
        x0 = int(np.clip(x, 1, W-2))
        y0 = int(np.clip(y, 1, H-2))

        dx = quad_offset(h[y0, x0-1], h[y0, x0], h[y0, x0+1])
        dy = quad_offset(h[y0-1, x0], h[y0, x0], h[y0+1, x0])

        pts[k] = [x0 + np.clip(dx, -0.5, 0.5), y0 + np.clip(dy, -0.5, 0.5)]

    return pts, conf


# -------------------------
# ROI crop helpers
# -------------------------
@dataclass
class CropInfo:
    x1: int
    y1: int
    x2: int
    y2: int
    scale_x: float
    scale_y: float


def compute_roi(points_xy: np.ndarray, visible: np.ndarray, w: int, h: int, margin: float = 1.6) -> tuple[int,int,int,int]:
    pts = points_xy[visible]
    if pts.size == 0:
        return 0, 0, w, h

    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)

    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    bw = (x_max - x_min) * margin
    bh = (y_max - y_min) * margin

    # if points are all same (bw=0), give a minimum box
    bw = max(bw, 64.0)
    bh = max(bh, 64.0)

    x1 = int(np.floor(cx - bw/2))
    y1 = int(np.floor(cy - bh/2))
    x2 = int(np.ceil (cx + bw/2))
    y2 = int(np.ceil (cy + bh/2))

    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)

    # ensure non-empty
    if x2 <= x1: x2 = min(w, x1 + 1)
    if y2 <= y1: y2 = min(h, y1 + 1)

    return x1, y1, x2, y2


def crop_and_resize(frame_rgb: np.ndarray, x1: int, y1: int, x2: int, y2: int, out_size: int) -> tuple[np.ndarray, CropInfo]:
    crop = frame_rgb[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    resized = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    scale_x = out_size / max(cw, 1)
    scale_y = out_size / max(ch, 1)
    return resized, CropInfo(x1=x1, y1=y1, x2=x2, y2=y2, scale_x=scale_x, scale_y=scale_y)


def points_to_crop(points_xy: np.ndarray, crop: CropInfo) -> np.ndarray:
    pts = points_xy.copy().astype(np.float32)
    pts[:, 0] = (pts[:, 0] - crop.x1) * crop.scale_x
    pts[:, 1] = (pts[:, 1] - crop.y1) * crop.scale_y
    return pts


def points_from_crop(points_xy_crop: np.ndarray, crop: CropInfo) -> np.ndarray:
    pts = points_xy_crop.copy().astype(np.float32)
    pts[:, 0] = pts[:, 0] / crop.scale_x + crop.x1
    pts[:, 1] = pts[:, 1] / crop.scale_y + crop.y1
    return pts


# -------------------------
# Dataset (ROI-cropped)
# -------------------------
class RoiHeatmapDataset(Dataset):
    def __init__(
        self,
        video_path: Path,
        labels_npz: Path,
        img_size: int = 512,
        hm_size: int = 128,
        sigma: float = 2.5,
        roi_margin: float = 1.6,
        max_labeled_frames: int | None = None,
        cache: bool = True,
    ):
        self.video_path = str(video_path)
        self.img_size = int(img_size)
        self.hm_size = int(hm_size)
        self.sigma = float(sigma)
        self.roi_margin = float(roi_margin)

        z = np.load(labels_npz, allow_pickle=True)
        self.points = z["points"].astype(np.float32)          # (T,K,2)
        self.visibility = z["visibility"].astype(np.uint8)    # (T,K) 0/1
        self.marker_names = list(z["marker_names"])
        self.K = self.points.shape[1]

        labeled = np.where(self.visibility.sum(axis=1) > 0)[0].astype(int)
        if max_labeled_frames is not None:
            labeled = labeled[:max_labeled_frames]
        self.frames = labeled

        self.cache = cache
        self.frame_cache = {}
        if cache:
            print("Caching labeled frames...")
            cap = cv2.VideoCapture(self.video_path)
            for fi in self.frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
                ok, frame = cap.read()
                if ok:
                    self.frame_cache[int(fi)] = frame
            cap.release()
            print(f"  Cached {len(self.frame_cache)} frames")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx: int):
        fi = int(self.frames[idx])

        if self.cache:
            frame_bgr = self.frame_cache[fi]
        else:
            cap = cv2.VideoCapture(self.video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame_bgr = cap.read()
            cap.release()
            if not ok:
                raise RuntimeError(f"Failed to read frame {fi}")

        h0, w0 = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        pts = self.points[fi].copy()
        vis = (self.visibility[fi] > 0)

        # ROI around visible points
        x1, y1, x2, y2 = compute_roi(pts, vis, w0, h0, margin=self.roi_margin)
        roi_rgb, cropinfo = crop_and_resize(frame_rgb, x1, y1, x2, y2, self.img_size)

        # normalize
        x = roi_rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], np.float32)
        std  = np.array([0.229, 0.224, 0.225], np.float32)
        x = (x - mean) / std
        x = torch.from_numpy(x).permute(2, 0, 1).float()  # (3,img,img)

        # points into crop coords then into heatmap coords
        pts_crop = points_to_crop(pts, cropinfo)                 # (K,2) in img coords
        pts_hm = pts_crop * (self.hm_size / self.img_size)       # (K,2) in hm coords

        y = make_gaussian_heatmaps(pts_hm, vis, self.hm_size, self.hm_size, sigma=self.sigma)
        y = torch.from_numpy(y).float()
        v = torch.from_numpy(vis.astype(np.float32))

        # return cropinfo for potential debugging (not used by training loop)
        return x, y, v


# -------------------------
# Model
# -------------------------
class HeatmapNet(nn.Module):
    def __init__(self, K: int):
        super().__init__()
        resnet = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.head = nn.Sequential(
            nn.ConvTranspose2d(2048, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, K, 1),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


# -------------------------
# Train
# -------------------------
def train(
    video_path: Path,
    labels_npz: Path,
    out_ckpt: Path,
    img_size: int = 512,
    hm_size: int = 128,
    sigma: float = 3.0,
    roi_margin: float = 1.6,
    epochs: int = 80,
    batch_size: int = 8,
    lr: float = 2e-4,
    device: str | None = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    print("Device:", device)

    ds = RoiHeatmapDataset(
        video_path=video_path,
        labels_npz=labels_npz,
        img_size=img_size,
        hm_size=hm_size,
        sigma=sigma,
        roi_margin=roi_margin,
        max_labeled_frames=None,
        cache=True,
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    model = HeatmapNet(K=ds.K).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0

        for x, y, v in dl:
            x, y, v = x.to(device), y.to(device), v.to(device)
            logits = model(x)

            if logits.shape[-1] != hm_size or logits.shape[-2] != hm_size:
                raise RuntimeError(
                    f"Model output is {tuple(logits.shape[-2:])} but hm_size={hm_size}. "
                    f"hm_size must equal img_size/4 for this model."
                )

            vmask = v.unsqueeze(-1).unsqueeze(-1)
            loss_map = F.binary_cross_entropy_with_logits(logits, y, reduction="none")

            # stronger peak weighting than before
            w = 1.0 + 25.0 * y
            loss = (loss_map * w * vmask).sum() / (vmask.sum() * hm_size * hm_size + 1e-8)

            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()

        if epoch == 1 or epoch % 5 == 0:
            print(f"epoch {epoch:03d}  loss {running/len(dl):.6f}")

    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "marker_names": ds.marker_names,
            "img_size": img_size,
            "hm_size": hm_size,
            "sigma": sigma,
            "roi_margin": roi_margin,
        },
        out_ckpt,
    )
    print("Saved ckpt:", out_ckpt)


# -------------------------
# Track video (ROI around previous prediction)
# -------------------------
def track_video(
    video_path: Path,
    ckpt_path: Path,
    out_video_path: Path,
    max_frames: int | None = 2000,
    device: str | None = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    print("Device:", device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    img_size = int(ckpt["img_size"])
    hm_size  = int(ckpt["hm_size"])
    sigma    = float(ckpt.get("sigma", 3.0))
    roi_margin = float(ckpt.get("roi_margin", 1.6))
    marker_names = ckpt["marker_names"]

    model = HeatmapNet(K=len(marker_names)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    cap = cv2.VideoCapture(str(video_path))
    ok, first = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read video: {video_path}")
    h0, w0 = first.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    out_video_path = Path(out_video_path)
    out_video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w0, h0))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    prev_pts = None
    all_pts, all_conf = [], []
    fi = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if max_frames is not None and fi >= max_frames:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Choose ROI: if we have prev prediction, crop around it; else use full frame
        if prev_pts is None:
            x1, y1, x2, y2 = 0, 0, w0, h0
        else:
            # treat all points visible for ROI purposes
            vis = np.ones((prev_pts.shape[0],), dtype=bool)
            x1, y1, x2, y2 = compute_roi(prev_pts, vis, w0, h0, margin=roi_margin)

        roi_rgb, cropinfo = crop_and_resize(frame_rgb, x1, y1, x2, y2, img_size)

        x = roi_rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], np.float32)
        std  = np.array([0.229, 0.224, 0.225], np.float32)
        x = (x - mean) / std
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().to(device)

        with torch.no_grad():
            logits = model(x)[0]
            hm = torch.sigmoid(logits)
            pts_hm, conf = decode_argmax_subpixel(hm)

        pts_crop = pts_hm * (img_size / hm_size)
        pts_orig = points_from_crop(pts_crop, cropinfo)

        # confidence gate + smoothing
        if prev_pts is not None:
            gate = conf < 0.25
            pts_orig[gate] = prev_pts[gate]
            alpha = 0.7
            pts_orig = alpha * prev_pts + (1 - alpha) * pts_orig

        prev_pts = pts_orig.copy()
        all_pts.append(pts_orig)
        all_conf.append(conf)

        vis_frame = frame_bgr.copy()
        for (xk, yk), ck in zip(pts_orig, conf):
            xk = int(np.clip(xk, 0, w0 - 1))
            yk = int(np.clip(yk, 0, h0 - 1))
            r = 6 if ck > 0.25 else 3
            cv2.circle(vis_frame, (xk, yk), r, (0, 0, 255), -1)

        cv2.putText(vis_frame, f"Frame {fi} mean_conf={float(np.mean(conf)):.2f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        writer.write(vis_frame)

        if fi % 100 == 0:
            print("frame", fi, "mean conf", float(np.mean(conf)))

        fi += 1

    cap.release()
    writer.release()

    out_npz = out_video_path.with_suffix(".tracks.npz")
    np.savez(out_npz, points=np.stack(all_pts), confidence=np.stack(all_conf), marker_names=marker_names)
    print("Saved:", out_video_path)
    print("Saved:", out_npz)


# -------------------------
# Main: your exact paths
# -------------------------
if __name__ == "__main__":
    path_to_data = Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1")

    video_path = path_to_data / "synchronized_videos" / "sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1_synced_Cam6.mp4"
    labels_path = path_to_data / "synchronized_videos" / "sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1_synced_Cam6.labels.npz"

    ckpt_path = path_to_data / "runs" / "roi_heatmap_tracker_r50.pt"
    out_video = path_to_data / "runs" / "roi_heatmap_tracker_r50_output.mp4"

    train(
        video_path=video_path,
        labels_npz=labels_path,
        out_ckpt=ckpt_path,
        img_size=512,
        hm_size=128,
        sigma=3.0,
        roi_margin=1.7,
        epochs=50,
        batch_size=16,
        lr=2e-4,
    )

    track_video(
        video_path=video_path,
        ckpt_path=ckpt_path,
        out_video_path=out_video,
        max_frames=2000,
    )
