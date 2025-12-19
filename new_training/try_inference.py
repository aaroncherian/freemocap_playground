import cv2
import numpy as np
import torch
from pathlib import Path
from new_training.try_training import ResNetPoseNet

def pad_to_square(img, pts=None):
    """Pad image to square, adjust points if provided."""
    h, w = img.shape[:2]
    size = max(h, w)
    pad_top = (size - h) // 2
    pad_left = (size - w) // 2
    
    padded = cv2.copyMakeBorder(
        img, pad_top, size - h - pad_top, pad_left, size - w - pad_left,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    
    if pts is not None:
        pts = pts.copy()
        pts[:, 0] += pad_left
        pts[:, 1] += pad_top
        return padded, pts
    return padded

def heatmap_argmax_xy(hm: torch.Tensor) -> np.ndarray:
    K, H, W = hm.shape
    flat = hm.view(K, -1)
    idx = flat.argmax(dim=1)
    y = (idx // W).float()
    x = (idx % W).float()
    return torch.stack([x, y], dim=1).cpu().numpy()


def subpixel_argmax(hm: torch.Tensor) -> np.ndarray:
    """Refine heatmap peak using quadratic fitting."""
    K, H, W = hm.shape
    hm_np = hm.cpu().numpy()
    pts = []
    
    for k in range(K):
        h = hm_np[k]
        idx = h.argmax()
        y, x = divmod(idx, W)
        
        if 1 <= x < W - 1 and 1 <= y < H - 1:
            dx = (h[y, x+1] - h[y, x-1]) / (2 * (2*h[y,x] - h[y,x-1] - h[y,x+1]) + 1e-6)
            dy = (h[y+1, x] - h[y-1, x]) / (2 * (2*h[y,x] - h[y-1,x] - h[y+1,x]) + 1e-6)
            dx = np.clip(dx, -0.5, 0.5)
            dy = np.clip(dy, -0.5, 0.5)
            x, y = x + dx, y + dy
            
        pts.append([float(x), float(y)])
    
    return np.array(pts, dtype=np.float32)

def draw_points(frame_bgr, pts_xy, color, radius=4, thickness=-1):
    for (x, y) in pts_xy:
        cv2.circle(frame_bgr, (int(x), int(y)), radius, color, thickness)
    return frame_bgr


def make_overlay_video(
    ckpt_path: Path,
    video_path: Path,
    labels_path: Path,
    out_video_path: Path,
    n_frames: int | None = None,
    start_frame: int = 0,
    device: str | None = None,
    save_predictions_npz: Path | None = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    # Load model checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    marker_names = ckpt["marker_names"]
    img_size = int(ckpt["img_size"])
    hm_size = int(ckpt["hm_size"])

    model = ResNetPoseNet(
        K=len(marker_names),
        hm_size=hm_size,
        backbone="resnet18",
        pretrained=False,   # we're loading weights, so don't init pretrained
    ).to(device)

    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    # Load labels (partial)
    z = np.load(labels_path, allow_pickle=True)
    points = z["points"]          # (T_lab,K,2) in original coords
    visibility = z["visibility"]  # (T_lab,K)

    T_lab = points.shape[0]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
    ok, frame0 = cap.read()
    if not ok:
        raise RuntimeError("Could not read start frame")
    H, W = frame0.shape[:2]
    padded_size = max(H, W)
    pad_top = (padded_size - H) // 2
    pad_left = (padded_size - W) // 2


    # Decide how many frames to process
    if n_frames is None:
        end_frame = total_video
    else:
        end_frame = min(total_video, start_frame + int(n_frames))

    # Video writer at original resolution
    out_video_path = Path(out_video_path)
    out_video_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (W, H))

    # rewind after probing first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))

    # optional: store predictions for whole run
    preds_all = []
    frame_ids = []
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
    for fi in range(start_frame, end_frame):
        ok, frame = cap.read()
        if not ok:
            break

        # Pad to square THEN resize (must match training)
        frame_padded = pad_to_square(frame)
        frame_rs = cv2.resize(frame_padded, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        
        rgb = cv2.cvtColor(frame_rs, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
        x = (x - mean) / std

        with torch.no_grad():
            hm = model(x)[0]

        if fi % 100 == 0:  # every 100th frame
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, len(marker_names) + 1, figsize=(15, 3))
            axes[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            axes[0].set_title("Frame")
            save_path = Path(f"debug_heatmaps")
            save_path.mkdir(parents=True, exist_ok=True)
            for k in range(len(marker_names)):
                axes[k+1].imshow(hm[k].cpu().numpy())
                axes[k+1].set_title(marker_names[k])

            plt.savefig(save_path/f"heatmap_{fi:04d}.png")
            plt.close()

        pred_xy_hm = subpixel_argmax(hm)

        pred_xy = pred_xy_hm * (padded_size / hm_size)  # heatmap to padded coords
        pred_xy[:, 0] -= pad_left  # remove padding offset
        pred_xy[:, 1] -= pad_top
        preds_all.append(pred_xy.astype(np.float32))
        frame_ids.append(fi)

        vis_frame = frame.copy()

        # Draw predictions always (red)
        draw_points(vis_frame, pred_xy, (0, 0, 255), radius=4, thickness=-1)

        # Draw GT only if this frame exists in labels
        if fi < T_lab:
            vis = visibility[fi] > 0.5
            gt = points[fi].astype(np.float32)
            draw_points(vis_frame, gt[vis], (0, 255, 0), radius=3, thickness=-1)

        cv2.putText(
            vis_frame,
            f"frame {fi} | GT: {'yes' if fi < T_lab else 'no'}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        writer.write(vis_frame)

    writer.release()
    cap.release()
    print(f"Saved overlay video: {out_video_path}")

    preds_all = np.stack(preds_all, axis=0) if len(preds_all) else np.zeros((0, len(marker_names), 2), np.float32)
    frame_ids = np.array(frame_ids, dtype=np.int32)

    if save_predictions_npz is not None:
        save_predictions_npz = Path(save_predictions_npz)
        save_predictions_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            save_predictions_npz,
            pred_xy=preds_all,              # (T_run,K,2) in original pixel coords
            frame_indices=frame_ids,        # frames these correspond to
            marker_names=np.array(marker_names),
            video_path=str(video_path),
        )
        print(f"Saved predictions npz: {save_predictions_npz}")


if __name__ == "__main__":

    path_to_data = Path(r"D:\sfn\michael_wobble\recording_12_07_09_gmt-5__MDN_wobble_3")

    make_overlay_video(
    ckpt_path=path_to_data / "runs/v2_cam6_resnet18.pt",
    video_path=path_to_data / "synchronized_videos" / "Camera_000_synchronized.mp4",
    labels_path=path_to_data / "synchronized_videos" / "Camera_000_synchronized.labels.npz",
    out_video_path=path_to_data / "runs/v2_cam6_resnet18.mp4",
    n_frames=None,
    start_frame=0,
)