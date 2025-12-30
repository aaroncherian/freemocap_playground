"""
Hybrid TAPIR + Learned Refinement

1. TAPIR provides coarse tracking (frozen, no training)
2. Small CNN refines predictions using local patches
3. Train refinement on your labeled data

Install:
    pip install git+https://github.com/google-deepmind/tapnet.git
    pip install tensorflow tensorflow-datasets
"""

from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import urllib.request

from tapnet.torch import tapir_model


# -------------------------
# TAPIR Wrapper (frozen)
# -------------------------
class TAPIRTracker:
    """Frozen TAPIR for coarse tracking."""
    
    def __init__(self, device=None, resize_height=256):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.resize_height = resize_height
        
        # Download checkpoint
        checkpoint_url = "https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt"
        checkpoint_path = Path.home() / ".cache" / "tapnet" / "bootstapir_checkpoint_v2.pt"
        
        if not checkpoint_path.exists():
            print("Downloading TAPIR checkpoint...")
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(checkpoint_url, checkpoint_path)
        
        self.model = tapir_model.TAPIR(pyramid_level=1)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device, weights_only=False))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Freeze all parameters
        for p in self.model.parameters():
            p.requires_grad = False
        
        print(f"TAPIR loaded on {self.device}")
    
    def track(self, frames, query_points):
        """
        Args:
            frames: (T, H, W, 3) float32 [0, 1]
            query_points: (N, 3) with [frame_idx, y, x] in original coords
        
        Returns:
            tracks: (T, N, 2) as [x, y] in original coords
            visibles: (T, N) visibility scores
        """
        T, H, W, _ = frames.shape
        
        # Resize for model
        scale = self.resize_height / H
        new_W = int(W * scale)
        frames_resized = np.stack([cv2.resize(f, (new_W, self.resize_height)) for f in frames])
        
        # Scale query points
        query_scaled = query_points.copy()
        query_scaled[:, 1] *= scale  # y
        query_scaled[:, 2] *= scale  # x
        
        # To tensors
        video_tensor = torch.from_numpy(frames_resized).unsqueeze(0).to(self.device)
        query_tensor = torch.from_numpy(query_scaled).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(video_tensor, query_tensor)
        
        tracks_raw = outputs["tracks"][0].cpu().numpy()
        occlusion_raw = outputs["occlusion"][0].cpu().numpy()
        
        # Handle shape (N, T, 2) -> (T, N, 2)
        if tracks_raw.shape[0] == len(query_points):
            tracks = np.transpose(tracks_raw, (1, 0, 2))
            occlusion = np.transpose(occlusion_raw, (1, 0))
        else:
            tracks = tracks_raw
            occlusion = occlusion_raw
        
        # Scale back to original resolution
        tracks[:, :, 0] /= scale
        tracks[:, :, 1] /= scale
        
        visibles = 1.0 / (1.0 + np.exp(occlusion))
        
        return tracks, visibles


# -------------------------
# Refinement Network
# -------------------------
class RefinementCNN(nn.Module):
    """
    Small CNN that looks at a patch around the coarse prediction
    and outputs a refined offset (dx, dy).
    """
    
    def __init__(self, patch_size=64):
        super().__init__()
        self.patch_size = patch_size
        
        # Simple ConvNet
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),  # 64 -> 32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 16 -> 8
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),  # 8 -> 4
            nn.ReLU(inplace=True),
        )
        
        # Output: offset (dx, dy) relative to patch center
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 2),  # (dx, dy)
        )
    
    def forward(self, patches):
        """
        Args:
            patches: (B, 3, patch_size, patch_size)
        Returns:
            offsets: (B, 2) predicted offset from patch center
        """
        f = self.features(patches)
        return self.regressor(f)


# -------------------------
# Dataset for Refinement Training
# -------------------------
class RefinementDataset(Dataset):
    """
    Dataset for training the refinement network.
    
    For each labeled point:
    1. Get TAPIR's coarse prediction (or simulate noise for augmentation)
    2. Extract patch around coarse prediction
    3. Target = offset from patch center to true label
    """
    
    def __init__(
        self,
        frames,           # (T, H, W, 3) float32 [0, 1]
        labels,           # (T, K, 2) ground truth points
        visibility,       # (T, K) visibility
        tapir_tracks,     # (T, K, 2) TAPIR predictions, or None to simulate
        patch_size=64,
        augment=True,
        noise_std=5.0,    # pixels, for simulating coarse error during training
    ):
        self.frames = frames
        self.labels = labels
        self.visibility = visibility
        self.tapir_tracks = tapir_tracks
        self.patch_size = patch_size
        self.augment = augment
        self.noise_std = noise_std
        
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        # Build index of valid (frame, keypoint) pairs
        self.samples = []
        T, K, _ = labels.shape
        for t in range(T):
            for k in range(K):
                if visibility[t, k] > 0.5:
                    self.samples.append((t, k))
        
        print(f"RefinementDataset: {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        t, k = self.samples[idx]
        
        frame = self.frames[t]  # (H, W, 3)
        H, W, _ = frame.shape
        
        gt_xy = self.labels[t, k]  # ground truth
        
        # Get coarse prediction (TAPIR or simulated)
        if self.tapir_tracks is not None:
            coarse_xy = self.tapir_tracks[t, k].copy()
        else:
            coarse_xy = gt_xy.copy()
        
        # Augmentation: add noise to coarse prediction
        if self.augment:
            noise = np.random.randn(2) * self.noise_std
            coarse_xy = coarse_xy + noise
        
        # Extract patch centered on coarse prediction
        cx, cy = int(coarse_xy[0]), int(coarse_xy[1])
        half = self.patch_size // 2
        
        # Compute crop bounds with padding handling
        x1, y1 = cx - half, cy - half
        x2, y2 = cx + half, cy + half
        
        # Pad frame if needed
        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - W)
        pad_bottom = max(0, y2 - H)
        
        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            frame_padded = np.pad(
                frame,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode='constant',
                constant_values=0
            )
            x1 += pad_left
            y1 += pad_top
            x2 += pad_left
            y2 += pad_top
        else:
            frame_padded = frame
        
        patch = frame_padded[y1:y2, x1:x2]  # (patch_size, patch_size, 3)
        
        # Target: offset from patch center to ground truth
        # Patch center in original coords = coarse_xy
        offset = gt_xy - coarse_xy  # (dx, dy)
        
        # To tensor
        patch_tensor = torch.from_numpy(patch).permute(2, 0, 1)  # (3, H, W)
        patch_tensor = (patch_tensor - self.mean) / self.std
        offset_tensor = torch.from_numpy(offset.astype(np.float32))
        
        return patch_tensor, offset_tensor


# -------------------------
# Training
# -------------------------
def train_refinement(
    video_path,
    labels_path,
    out_path,
    tapir_tracks=None,     # Optional: precomputed TAPIR tracks
    patch_size=64,
    epochs=50,
    batch_size=32,
    lr=1e-3,
    noise_std=5.0,
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    
    # Load video
    print(f"Loading video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    frames = np.stack(frames).astype(np.float32) / 255.0
    T, H, W, _ = frames.shape
    print(f"  {T} frames, {W}x{H}")
    
    # Load labels
    z = np.load(labels_path, allow_pickle=True)
    points = z["points"].astype(np.float32)
    visibility = z["visibility"]
    marker_names = list(z["marker_names"])
    K = len(marker_names)
    print(f"  {K} markers: {marker_names}")
    
    # Create dataset
    ds = RefinementDataset(
        frames=frames,
        labels=points,
        visibility=visibility,
        tapir_tracks=tapir_tracks,
        patch_size=patch_size,
        augment=True,
        noise_std=noise_std,
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Model
    model = RefinementCNN(patch_size=patch_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    
    print(f"Training refinement network...")
    print(f"  patch_size={patch_size}, noise_std={noise_std}")
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        
        for patches, offsets in dl:
            patches = patches.to(device)
            offsets = offsets.to(device)
            
            pred = model(patches)
            loss = F.mse_loss(pred, offsets)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            running_loss += loss.item()
        
        scheduler.step()
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | loss {running_loss/len(dl):.4f}")
    
    # Save
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "patch_size": patch_size,
        "marker_names": marker_names,
    }, out_path)
    print(f"Saved refinement model: {out_path}")
    
    return model


# -------------------------
# Full Pipeline: TAPIR + Refinement
# -------------------------
class HybridTracker:
    """Combined TAPIR (coarse) + Refinement (precise) tracker."""
    
    def __init__(self, refinement_ckpt_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load TAPIR
        print("Loading TAPIR...")
        self.tapir = TAPIRTracker(device=self.device)
        
        # Load refinement
        print(f"Loading refinement: {refinement_ckpt_path}")
        ckpt = torch.load(refinement_ckpt_path, map_location=self.device, weights_only=False)
        self.patch_size = ckpt["patch_size"]
        self.marker_names = ckpt["marker_names"]
        
        self.refine = RefinementCNN(patch_size=self.patch_size).to(self.device)
        self.refine.load_state_dict(ckpt["state_dict"])
        self.refine.eval()
        
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        
        print("Hybrid tracker ready.")
    
    def extract_patches(self, frame, points, patch_size):
        """Extract patches around each point."""
        H, W, _ = frame.shape
        half = patch_size // 2
        patches = []
        
        for (x, y) in points:
            cx, cy = int(x), int(y)
            x1, y1 = cx - half, cy - half
            x2, y2 = cx + half, cy + half
            
            pad_left = max(0, -x1)
            pad_top = max(0, -y1)
            pad_right = max(0, x2 - W)
            pad_bottom = max(0, y2 - H)
            
            if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
                frame_padded = np.pad(
                    frame,
                    ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                    mode='constant'
                )
                patch = frame_padded[y1+pad_top:y2+pad_top, x1+pad_left:x2+pad_left]
            else:
                patch = frame[y1:y2, x1:x2]
            
            patches.append(patch)
        
        return np.stack(patches)  # (N, patch_size, patch_size, 3)
    
    def track(self, frames, query_points):
        """
        Full tracking pipeline.
        
        Args:
            frames: (T, H, W, 3) float32 [0, 1]
            query_points: (N, 3) with [frame_idx, y, x]
        
        Returns:
            refined_tracks: (T, N, 2)
            tapir_tracks: (T, N, 2)
            visibles: (T, N)
        """
        # Step 1: TAPIR coarse tracking
        tapir_tracks, visibles = self.tapir.track(frames, query_points)
        
        # Step 2: Refine each frame
        T, N, _ = tapir_tracks.shape
        refined_tracks = np.zeros_like(tapir_tracks)
        
        for t in range(T):
            frame = frames[t]
            coarse = tapir_tracks[t]  # (N, 2)
            
            patches = self.extract_patches(frame, coarse, self.patch_size)
            patches_tensor = torch.from_numpy(patches).permute(0, 3, 1, 2).float().to(self.device)
            patches_tensor = (patches_tensor - self.mean) / self.std
            
            with torch.no_grad():
                offsets = self.refine(patches_tensor).cpu().numpy()
            
            refined_tracks[t] = coarse + offsets
        
        return refined_tracks, tapir_tracks, visibles


# -------------------------
# Inference with overlay video
# -------------------------
def run_hybrid_tracking(
    video_path,
    labels_path,
    refinement_ckpt_path,
    out_video_path,
    query_frame=0,
    max_frames=None,
):
    video_path = Path(video_path)
    labels_path = Path(labels_path)
    out_video_path = Path(out_video_path)
    
    # Load video
    print(f"Loading video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    frames = np.stack(frames).astype(np.float32) / 255.0
    T, H, W, _ = frames.shape
    print(f"  {T} frames, {W}x{H}")
    
    # Load query points
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
            query_points.append([query_frame, pts[k, 1], pts[k, 0]])  # [t, y, x]
            valid_indices.append(k)
    query_points = np.array(query_points, dtype=np.float32)
    
    print(f"  Tracking {len(query_points)} points")
    
    # Initialize tracker
    tracker = HybridTracker(refinement_ckpt_path)
    
    # Run tracking
    print("Running hybrid tracking...")
    refined, coarse, visibles = tracker.track(frames, query_points)
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
    
    for t in range(T):
        frame_bgr = (frames[t, :, :, ::-1] * 255).astype(np.uint8).copy()
        
        for n in range(len(valid_indices)):
            # Coarse (small, hollow)
            cx, cy = coarse[t, n]
            color = colors[n % len(colors)]
            cv2.circle(frame_bgr, (int(cx), int(cy)), 3, color, 1)
            
            # Refined (larger, filled)
            rx, ry = refined[t, n]
            cv2.circle(frame_bgr, (int(rx), int(ry)), 6, color, -1)
            cv2.circle(frame_bgr, (int(rx), int(ry)), 6, (255, 255, 255), 1)
            
            label = marker_names[valid_indices[n]]
            cv2.putText(frame_bgr, label, (int(rx) + 8, int(ry) - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        cv2.putText(frame_bgr, f"Frame {t} | hollow=TAPIR, filled=refined", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        writer.write(frame_bgr)
    
    writer.release()
    print(f"Saved: {out_video_path}")
    
    # Save tracks
    tracks_path = out_video_path.with_suffix(".tracks.npz")
    np.savez(
        tracks_path,
        refined_tracks=refined,
        tapir_tracks=coarse,
        visibles=visibles,
        marker_names=np.array([marker_names[i] for i in valid_indices]),
    )
    print(f"Saved: {tracks_path}")


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    path_to_data = Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1")
    
    # # Step 1: Train refinement network
    train_refinement(
        video_path=path_to_data / "synchronized_videos" / "sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1_synced_Cam6.mp4",
        labels_path=path_to_data / "synchronized_videos" / "sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1_synced_Cam6.labels.npz",
        out_path=path_to_data / "runs/refinement_model.pt",
        patch_size=64,
        epochs=50,
        noise_std=5.0,
    )
    
    # Step 2: Run hybrid tracking
    run_hybrid_tracking(
        video_path=path_to_data / "synchronized_videos" / "sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1_synced_Cam6.mp4",
        labels_path=path_to_data / "synchronized_videos" / "sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1_synced_Cam6.labels.npz",
        refinement_ckpt_path=path_to_data / "runs/refinement_model.pt",
        out_video_path=path_to_data / "runs/hybrid_tracking.mp4",
        query_frame=0,
        max_frames=2000,
    )