"""
TAPIR point tracking - zero-shot, no training needed.
Give it points on one frame, it tracks through the video.

Install:
    pip install git+https://github.com/google-deepmind/tapnet.git
    pip install tensorflow tensorflow-datasets
"""

from pathlib import Path
import numpy as np
import cv2
import torch
import urllib.request

from tapnet.torch import tapir_model


def download_checkpoint(url, dest):
    """Download checkpoint if not exists."""
    dest = Path(dest)
    if dest.exists():
        print(f"  Checkpoint exists: {dest}")
        return dest
    
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading checkpoint...")
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved to: {dest}")
    return dest


def load_video_frames(video_path, max_frames=None):
    """Load video as numpy array (T, H, W, 3) RGB float32 [0,1]."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        if max_frames and len(frames) >= max_frames:
            break
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    
    video = np.stack(frames).astype(np.float32) / 255.0
    return video, fps


def load_query_points(labels_path, query_frame=0):
    """Load points from your existing labels."""
    z = np.load(labels_path, allow_pickle=True)
    points = z["points"]
    visibility = z["visibility"]
    marker_names = list(z["marker_names"])
    
    vis = visibility[query_frame] > 0
    pts = points[query_frame]
    
    query_points = []
    valid_indices = []
    for k in range(len(pts)):
        if vis[k]:
            # TAPIR format: [t, y, x]
            query_points.append([query_frame, pts[k, 1], pts[k, 0]])
            valid_indices.append(k)
    
    return np.array(query_points, dtype=np.float32), marker_names, valid_indices


def resize_video(frames, height=256):
    """Resize video to target height, maintaining aspect ratio."""
    T, H, W, C = frames.shape
    scale = height / H
    new_W = int(W * scale)
    
    resized = np.stack([
        cv2.resize(f, (new_W, height)) for f in frames
    ])
    return resized, scale


def run_tapir_tracking(
    video_path,
    labels_path,
    out_video_path,
    query_frame=0,
    max_frames=None,
    resize_height=256,
):
    video_path = Path(video_path)
    labels_path = Path(labels_path)
    out_video_path = Path(out_video_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load video
    print(f"Loading video: {video_path}")
    frames, fps = load_video_frames(video_path, max_frames=max_frames)
    T, H, W, _ = frames.shape
    print(f"  {T} frames, {W}x{H}, {fps:.1f} fps")
    
    # Resize for model
    frames_resized, scale = resize_video(frames, height=resize_height)
    _, H_rs, W_rs, _ = frames_resized.shape
    print(f"  Resized to {W_rs}x{H_rs} for model (scale={scale:.3f})")
    
    # Load query points
    print(f"Loading query points from frame {query_frame}")
    query_points, marker_names, valid_indices = load_query_points(labels_path, query_frame)
    
    # Scale query points to resized video
    query_points_scaled = query_points.copy()
    query_points_scaled[:, 1] *= scale  # y
    query_points_scaled[:, 2] *= scale  # x
    
    print(f"  Tracking {len(query_points)} points: {[marker_names[i] for i in valid_indices]}")
    
    # Download and load model
    print("Loading TAPIR model...")
    checkpoint_url = "https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt"
    checkpoint_path = Path.home() / ".cache" / "tapnet" / "bootstapir_checkpoint_v2.pt"
    download_checkpoint(checkpoint_url, checkpoint_path)
    
    model = tapir_model.TAPIR(pyramid_level=1)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    print("  Model loaded.")
    
    # Prepare inputs - TAPIR expects (B, T, H, W, 3)
    video_tensor = torch.from_numpy(frames_resized).unsqueeze(0).to(device)  # (1, T, H, W, 3)
    query_tensor = torch.from_numpy(query_points_scaled).unsqueeze(0).to(device)  # (1, N, 3)
    
    # Run tracking
    print("Running TAPIR tracking...")
    with torch.no_grad():
        outputs = model(video_tensor, query_tensor)
    
    # Check output shapes
    tracks_raw = outputs["tracks"][0].cpu().numpy()
    occlusion_raw = outputs["occlusion"][0].cpu().numpy()
    print(f"  tracks shape: {tracks_raw.shape}, occlusion shape: {occlusion_raw.shape}")
    
    # TAPIR returns (N, T, 2) - transpose to (T, N, 2)
    if tracks_raw.shape[0] == len(valid_indices):
        tracks = np.transpose(tracks_raw, (1, 0, 2))  # (N, T, 2) -> (T, N, 2)
        occlusion = np.transpose(occlusion_raw, (1, 0))  # (N, T) -> (T, N)
    else:
        tracks = tracks_raw
        occlusion = occlusion_raw
    
    print(f"  tracks (reordered): {tracks.shape}")
    
    # Convert occlusion logits to visibility probability
    visibles = 1.0 / (1.0 + np.exp(occlusion))  # sigmoid
    
    # Scale tracks back to original resolution
    tracks[:, :, 0] /= scale  # x
    tracks[:, :, 1] /= scale  # y
    
    print("  Done.")
    
    # Create overlay video
    print(f"Creating overlay video: {out_video_path}")
    out_video_path.parent.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (W, H))
    
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
    ]
    
    # Reload original video for overlay
    cap = cv2.VideoCapture(str(video_path))
    
    for t in range(T):
        ok, frame_bgr = cap.read()
        if not ok:
            break
        
        for n in range(len(valid_indices)):
            x, y = tracks[t, n]
            conf = visibles[t, n]
            color = colors[n % len(colors)]
            
            if conf > 0.5:
                cv2.circle(frame_bgr, (int(x), int(y)), 6, color, -1)
                cv2.circle(frame_bgr, (int(x), int(y)), 6, (255, 255, 255), 1)
            else:
                cv2.circle(frame_bgr, (int(x), int(y)), 6, color, 1)
            
            label = marker_names[valid_indices[n]]
            cv2.putText(frame_bgr, label, (int(x) + 8, int(y) - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        cv2.putText(frame_bgr, f"Frame {t}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        writer.write(frame_bgr)
    
    cap.release()
    writer.release()
    print(f"Saved: {out_video_path}")
    
    # Save tracks
    tracks_path = out_video_path.with_suffix(".tracks.npz")
    np.savez(
        tracks_path,
        tracks=tracks,
        visibles=visibles,
        marker_names=np.array([marker_names[i] for i in valid_indices]),
        query_frame=query_frame,
    )
    print(f"Saved tracks: {tracks_path}")
    
    return tracks, visibles


if __name__ == "__main__":
    path_to_data = Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1")
    
    run_tapir_tracking(
        video_path=path_to_data / "synchronized_videos" / "sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1_synced_Cam6.mp4",
        labels_path=path_to_data / "synchronized_videos" / "sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1_synced_Cam6.labels.npz",
        out_video_path=path_to_data / "runs/tapir_tracking.mp4",
        query_frame=0,
        max_frames=2000,
    )