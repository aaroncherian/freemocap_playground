from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn

from backbones import CSPNeXtBackbone
from heads import RTMCCHead
from new_training.rtmpose.codecs.simcc import SimCCCodec


class RTMPoseLike(nn.Module):
    def __init__(self, backbone_name: str, K: int, input_w: int, input_h: int, split_ratio: float):
        super().__init__()
        self.backbone = CSPNeXtBackbone(backbone_name, pretrained=False)
        self.head = RTMCCHead(self.backbone.out_dim, K, input_w, input_h, simcc_split_ratio=split_ratio)

    def forward(self, img):
        feat = self.backbone(img)
        return self.head(feat)


def draw_points(frame, xy, color=(0, 255, 0), vis=None):
    for i, (x, y) in enumerate(xy):
        if vis is not None and not bool(vis[i] > 0):
            continue
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        xi, yi = int(round(float(x))), int(round(float(y)))
        cv2.circle(frame, (xi, yi), 4, color, -1)
        cv2.putText(frame, str(i), (xi + 5, yi - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


@torch.no_grad()
def main(video_path: Path, labels_path: Path | None, ckpt_path: Path, out_video: Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    state = torch.load(ckpt_path, map_location=device)
    cfg = state["cfg"]

    K = int(cfg["K"])
    input_w = int(cfg["input_w"])
    input_h = int(cfg["input_h"])
    split_ratio = float(cfg["simcc_split_ratio"])
    backbone_name = str(cfg["backbone_name"])

    codec = SimCCCodec(
        input_w=input_w,
        input_h=input_h,
        simcc_split_ratio=split_ratio,
        sigma_x=float(cfg.get("sigma_x", 6.0)),
        sigma_y=float(cfg.get("sigma_y", 6.93)),
    )

    model = RTMPoseLike(backbone_name, K, input_w, input_h, split_ratio).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    # Optional GT overlay
    pts = vis = None
    if labels_path is not None and Path(labels_path).exists():
        npz = np.load(labels_path, allow_pickle=True)
        pts = npz["points"].astype(np.float32)      # (T,K,2) in original frame coords
        vis = npz["visibility"].astype(np.int32)    # (T,K)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    fidx = 0

    while True:
        ok, frame = cap.read()
        if fidx in (0, 1, 2, 10, 50, 100):
            print("frame", fidx,
                "mean", float(frame.mean()),
                "std", float(frame.std()),
                "sum", float(frame.sum()))
            
            if fidx == 0:
                prev = frame.copy()
            else:
                diff = np.mean(np.abs(frame.astype(np.float32) - prev.astype(np.float32)))
                if fidx in (1,2,10,50,100):
                    print("frame", fidx, "mean_abs_diff_from_prev", float(diff))
                prev = frame.copy()
        if not ok:
            break

        # FULL FRAME always
        frame_rs = cv2.resize(frame, (input_w, input_h), interpolation=cv2.INTER_LINEAR)

        frame_rgb = cv2.cvtColor(frame_rs, cv2.COLOR_BGR2RGB)

        img = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = (img - mean) / std
        img = img.unsqueeze(0).to(device)

        lx, ly = model(img)
        px = torch.softmax(lx[0, 0], dim=-1)  # keypoint 0
        py = torch.softmax(ly[0, 0], dim=-1)
        if fidx in (0, 1, 2, 10, 50, 100):
            print("kp0 px max", float(px.max()), "argmax", int(px.argmax()))
            print("kp0 py max", float(py.max()), "argmax", int(py.argmax()))
        if fidx in (0, 1, 2, 10, 50, 100):
            print("lx stats:", float(lx.mean()), float(lx.std()), float(lx.max()))
            print("ly stats:", float(ly.mean()), float(ly.std()), float(ly.max()))
        xy_in = codec.decode(lx, ly)[0].cpu().numpy()  # coords in resized frame space (input_w,input_h)

        # Map back to original frame coords
        sx = W / max(1, input_w)
        sy = H / max(1, input_h)
        xy = xy_in.copy()
        xy[:, 0] *= sx
        xy[:, 1] *= sy

        draw_points(frame, xy, (0, 255, 0))

        if pts is not None and fidx < pts.shape[0]:
            draw_points(frame, pts[fidx], (0, 0, 255), vis=vis[fidx])

        cv2.putText(frame, f"frame {fidx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        writer.write(frame)
        fidx += 1

    cap.release()
    writer.release()
    print("Wrote:", out_video)


if __name__ == "__main__":
    path_to_data = Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1")

    video_path  = path_to_data / "synchronized_videos" / "sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1_synced_Cam6.mp4"
    labels_path = path_to_data / "synchronized_videos" / "sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1_synced_Cam6.labels.npz"
    ckpt_path   = path_to_data / "runs" / "rtmpose_like_cspnext.pt"
    out_video   = path_to_data / "runs" / "fullframe_inference_overlay.mp4"

    main(video_path, labels_path, ckpt_path, out_video)
