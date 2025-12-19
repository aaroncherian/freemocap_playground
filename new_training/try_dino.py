"""
DINOv2 Point Tracker - First Pass Proof of Concept
Track arbitrary points through video using DINOv2 feature matching.

Usage:
    python dino_tracker.py path/to/video.mp4

Requirements:
    pip install torch torchvision opencv-python numpy

First run will download DINOv2 weights (~85MB).
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class DINOv2Tracker:
    """
    Track points through video using DINOv2 feature correspondence.
    
    No training required - works by finding locations in each frame
    that have similar features to your reference points.
    """
    
    def __init__(self, model_size: str = 'small', device: str = None):
        """
        Args:
            model_size: 'small', 'base', or 'large' (small is fastest)
            device: 'cuda', 'cpu', or None for auto-detect
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        print(f"Using device: {device}")
        
        # Model variants
        model_names = {
            'small': 'dinov2_vits14',  # 21M params, fastest
            'base': 'dinov2_vitb14',   # 86M params
            'large': 'dinov2_vitl14',  # 300M params, most accurate
        }
        
        feature_dims = {'small': 384, 'base': 768, 'large': 1024}
        
        print(f"Loading DINOv2-{model_size}...")
        self.model = torch.hub.load('facebookresearch/dinov2', model_names[model_size])
        self.model.eval().to(device)
        self.feature_dim = feature_dims[model_size]
        
        # DINOv2 uses 14x14 patches, input should be multiple of 14
        # 518 = 14 * 37 patches
        self.input_size = 518
        self.num_patches = 37
        
        # State
        self.reference_features = None
        self.query_features = None
        self.original_size = None
        
        # ImageNet normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    
    def set_reference(self, frame: np.ndarray, points: np.ndarray):
        """
        Set reference frame and points to track.
        
        Args:
            frame: BGR image (H, W, 3)
            points: (N, 2) array of (x, y) coordinates
        """
        self.original_size = (frame.shape[1], frame.shape[0])  # W, H
        
        # Extract features for reference frame
        tensor = self._preprocess(frame)
        self.reference_features = self._extract_features(tensor)
        
        # Sample features at query points
        self.query_features = []
        for x, y in points:
            feat = self._sample_feature_at_point(self.reference_features, x, y)
            self.query_features.append(feat)
        
        print(f"Initialized tracking for {len(points)} points")
    
    def track_frame(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Find tracked points in a new frame.
        
        Args:
            frame: BGR image (H, W, 3)
            
        Returns:
            points: (N, 2) array of (x, y) coordinates
            confidences: (N,) array of confidence scores (0-1)
        """
        tensor = self._preprocess(frame)
        features = self._extract_features(tensor)
        
        points = []
        confidences = []
        
        for query_feat in self.query_features:
            # Compute cosine similarity across spatial dimensions
            similarity = F.cosine_similarity(
                query_feat.view(1, -1, 1, 1),
                features,
                dim=1
            )  # (1, 37, 37)
            
            # Upsample similarity map for sub-patch precision
            similarity_up = F.interpolate(
                similarity.unsqueeze(0),
                size=(self.input_size, self.input_size),
                mode='bilinear',
                align_corners=False
            )[0, 0]  # (518, 518)
            
            # Find peak
            sim_np = similarity_up.cpu().numpy()
            max_val = sim_np.max()
            max_idx = np.unravel_index(np.argmax(sim_np), sim_np.shape)
            
            # Convert to original image coordinates
            y = max_idx[0] / self.input_size * self.original_size[1]
            x = max_idx[1] / self.input_size * self.original_size[0]
            
            points.append([x, y])
            confidences.append(max_val)
        
        return np.array(points), np.array(confidences)
    
    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """Convert BGR frame to normalized tensor."""
        img = cv2.resize(frame, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)
        tensor = (tensor - self.mean) / self.std
        
        return tensor
    
    @torch.no_grad()
    def _extract_features(self, tensor: torch.Tensor) -> torch.Tensor:
        """Extract spatial feature map from DINOv2."""
        features = self.model.forward_features(tensor)
        patches = features['x_norm_patchtokens']  # (1, 1369, feat_dim)
        
        # Reshape to spatial (1, feat_dim, 37, 37)
        patches = patches.permute(0, 2, 1).reshape(
            1, self.feature_dim, self.num_patches, self.num_patches
        )
        return patches
    
    def _sample_feature_at_point(self, features: torch.Tensor, x: float, y: float) -> torch.Tensor:
        """Get feature vector at image coordinate (x, y)."""
        # Convert to feature map coordinates
        fx = x / self.original_size[0] * self.num_patches
        fy = y / self.original_size[1] * self.num_patches
        
        # Bilinear interpolation for sub-pixel sampling
        fx_floor, fy_floor = int(fx), int(fy)
        fx_ceil = min(fx_floor + 1, self.num_patches - 1)
        fy_ceil = min(fy_floor + 1, self.num_patches - 1)
        
        # Interpolation weights
        wx = fx - fx_floor
        wy = fy - fy_floor
        
        # Sample four corners and interpolate
        f00 = features[0, :, fy_floor, fx_floor]
        f01 = features[0, :, fy_floor, fx_ceil]
        f10 = features[0, :, fy_ceil, fx_floor]
        f11 = features[0, :, fy_ceil, fx_ceil]
        
        feat = (
            f00 * (1 - wx) * (1 - wy) +
            f01 * wx * (1 - wy) +
            f10 * (1 - wx) * wy +
            f11 * wx * wy
        )
        
        return feat


def select_points(frame: np.ndarray, window_name: str = "Click points, press 'q' when done") -> np.ndarray:
    """Interactive point selection on a frame."""
    points = []
    display = frame.copy()
    
    def click_handler(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            cv2.circle(display, (x, y), 6, (0, 255, 0), -1)
            cv2.circle(display, (x, y), 8, (0, 0, 0), 2)
            cv2.imshow(window_name, display)
            print(f"  Point {len(points)}: ({x}, {y})")
    
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(1280, frame.shape[1]), min(720, frame.shape[0]))
    cv2.setMouseCallback(window_name, click_handler)
    cv2.imshow(window_name, display)
    
    print("Click to add points. Press 'q' when done, 'r' to reset.")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') and len(points) > 0:
            break
        elif key == ord('r'):
            points.clear()
            display = frame.copy()
            cv2.imshow(window_name, display)
            print("  Reset points")
    
    cv2.destroyAllWindows()
    return np.array(points)


def track_video(
    video_path: str,
    output_path: str = None,
    model_size: str = 'small',
    show_preview: bool = True
) -> np.ndarray:
    """
    Main function: track points through a video.
    
    Args:
        video_path: Input video file
        output_path: Optional output video with visualization
        model_size: 'small', 'base', or 'large'
        show_preview: Show tracking progress in window
        
    Returns:
        tracks: (T, N, 2) array of trajectories
    """
    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {total_frames} frames, {width}x{height}, {fps:.1f} fps")
    
    # Read first frame and select points
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Could not read first frame")
    
    print("\nSelect points to track:")
    points = select_points(first_frame)
    print(f"\nTracking {len(points)} points...")
    
    # Initialize tracker
    tracker = DINOv2Tracker(model_size=model_size)
    tracker.set_reference(first_frame, points)
    
    # Setup output video
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Colors for visualization
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
        (128, 255, 0), (0, 128, 255)
    ]
    
    # Track through video
    all_tracks = [points.copy()]
    all_confidences = [np.ones(len(points))]
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
    start_time = time.time()
    
    if show_preview:
        cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx == 0:
            # First frame uses initial points
            tracked_points = points.copy()
            confidences = np.ones(len(points))
        else:
            tracked_points, confidences = tracker.track_frame(frame)
        
        all_tracks.append(tracked_points.copy())
        all_confidences.append(confidences.copy())
        
        # Visualize
        vis = frame.copy()
        for i, (pt, conf) in enumerate(zip(tracked_points, confidences)):
            color = colors[i % len(colors)]
            x, y = int(pt[0]), int(pt[1])
            
            # Draw point
            cv2.circle(vis, (x, y), 6, color, -1)
            cv2.circle(vis, (x, y), 8, (0, 0, 0), 2)
            
            # Draw trail (last 15 frames)
            trail_start = max(0, len(all_tracks) - 15)
            for t in range(trail_start, len(all_tracks) - 1):
                pt1 = all_tracks[t][i]
                pt2 = all_tracks[t + 1][i]
                alpha = (t - trail_start) / 15
                cv2.line(vis, 
                        (int(pt1[0]), int(pt1[1])),
                        (int(pt2[0]), int(pt2[1])),
                        color, max(1, int(3 * alpha)))
        
        # Show stats
        elapsed = time.time() - start_time
        fps_actual = (frame_idx + 1) / elapsed if elapsed > 0 else 0
        cv2.putText(vis, f"Frame {frame_idx}/{total_frames} | {fps_actual:.1f} fps",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if output_path:
            out.write(vis)
        
        if show_preview:
            cv2.imshow("Tracking", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nTracking interrupted by user")
                break
        
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            print(f"  Frame {frame_idx}/{total_frames} ({fps_actual:.1f} fps)")
    
    cap.release()
    if output_path:
        out.release()
        print(f"\nSaved visualization to: {output_path}")
    
    if show_preview:
        cv2.destroyAllWindows()
    
    tracks = np.array(all_tracks)  # (T, N, 2)
    
    # Summary
    total_time = time.time() - start_time
    print(f"\nTracking complete:")
    print(f"  Frames: {len(tracks)}")
    print(f"  Time: {total_time:.1f}s ({len(tracks)/total_time:.1f} fps)")
    print(f"  Output shape: {tracks.shape}")
    
    return tracks


def main():
    parser = argparse.ArgumentParser(description="Track points through video using DINOv2")
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("-o", "--output", help="Path to output video (optional)")
    parser.add_argument("-m", "--model", choices=['small', 'base', 'large'], 
                        default='base', help="Model size (default: base)")
    parser.add_argument("--no-preview", action="store_true", help="Disable preview window")
    parser.add_argument("--save-tracks", help="Save tracks to .npy file")
    
    args = parser.parse_args()
    
    # Default output path
    if args.output is None:
        video_path = Path(args.video)
        args.output = str(video_path.parent / f"{video_path.stem}_tracked.mp4")
    
    # Run tracking
    tracks = track_video(
        args.video,
        output_path=args.output,
        model_size=args.model,
        show_preview=not args.no_preview
    )
    
    # Save tracks
    if args.save_tracks:
        np.save(args.save_tracks, tracks)
        print(f"Saved tracks to: {args.save_tracks}")
    else:
        # Default save location
        tracks_path = Path(args.video).with_suffix('.npy')
        np.save(tracks_path, tracks)
        print(f"Saved tracks to: {tracks_path}")


if __name__ == "__main__":
    main()