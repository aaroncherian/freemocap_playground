"""
CoTracker Point Tracker - Should actually work for gait!

CoTracker follows points through time using motion + learned features,
handles appearance changes, occlusion, fast motion.

Usage:
    python cotracker_tracker.py path/to/video.mp4

Requirements:
    pip install torch torchvision opencv-python numpy imageio[ffmpeg]
    
No separate CoTracker install needed - loads via torch.hub automatically.
First run will download model weights (~50MB).
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch


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


def load_video_frames(video_path: str, max_frames: int = None) -> tuple[np.ndarray, float]:
    """Load video as numpy array."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB for CoTracker
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if max_frames and len(frames) >= max_frames:
            break
    
    cap.release()
    return np.stack(frames), fps


def track_with_cotracker(
    video_path: str,
    output_path: str = None,
    show_preview: bool = True,
    max_frames_per_chunk: int = 100,
    resize_factor: float = 1.0,
    max_frames: int = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Track points through video using CoTracker.
    
    Processes video in chunks to avoid OOM errors.
    
    Args:
        video_path: Input video file
        output_path: Optional output video path
        show_preview: Show tracking progress
        max_frames_per_chunk: Process this many frames at a time (lower = less VRAM)
        resize_factor: Downscale video by this factor (0.5 = half resolution)
        max_frames: Only process first N frames (for testing)
    
    Returns:
        tracks: (T, N, 2) array of xy coordinates (in original resolution)
        visibility: (T, N) boolean array of whether point is visible
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load video
    print("Loading video...")
    frames, fps = load_video_frames(video_path, max_frames=max_frames)
    T, H, W, _ = frames.shape
    print(f"Video: {T} frames, {W}x{H}, {fps:.1f} fps")
    
    # Optionally resize to reduce memory
    if resize_factor != 1.0:
        new_H, new_W = int(H * resize_factor), int(W * resize_factor)
        print(f"Resizing to {new_W}x{new_H} to reduce memory...")
        frames_resized = np.stack([
            cv2.resize(f, (new_W, new_H)) for f in frames
        ])
        scale_x, scale_y = W / new_W, H / new_H
    else:
        frames_resized = frames
        new_H, new_W = H, W
        scale_x, scale_y = 1.0, 1.0
    
    # Select points on first frame (original resolution for display)
    print("\nSelect points to track:")
    first_frame_bgr = cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR)
    points = select_points(first_frame_bgr)
    N = len(points)
    
    # Scale points to resized resolution
    points_scaled = points.copy().astype(float)
    points_scaled[:, 0] /= scale_x
    points_scaled[:, 1] /= scale_y
    
    print(f"\nTracking {N} points with CoTracker...")
    print(f"Processing in chunks of {max_frames_per_chunk} frames")
    
    # Load CoTracker
    print("Loading CoTracker model...")
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker2").to(device)
    
    # Process in chunks with overlap
    overlap = 10  # Frames of overlap between chunks for continuity
    all_tracks = []
    all_visibility = []
    
    start_time = time.time()
    chunk_start = 0
    last_positions = points_scaled.copy()
    
    while chunk_start < T:
        chunk_end = min(chunk_start + max_frames_per_chunk, T)
        print(f"  Processing frames {chunk_start}-{chunk_end}...")
        
        # Get chunk
        chunk_frames = frames_resized[chunk_start:chunk_end]
        
        # Prepare video tensor: (1, chunk_T, 3, H, W)
        video_tensor = torch.from_numpy(chunk_frames).permute(0, 3, 1, 2).float()
        video_tensor = video_tensor.unsqueeze(0).to(device)
        
        # Prepare queries
        queries = torch.zeros(1, N, 3, device=device)
        queries[0, :, 0] = 0  # Query at frame 0 of this chunk
        queries[0, :, 1:] = torch.from_numpy(last_positions).float()
        
        # Run tracking
        with torch.no_grad():
            pred_tracks, pred_visibility = model(video_tensor, queries=queries)
        
        # Extract results
        chunk_tracks = pred_tracks[0].cpu().numpy()  # (chunk_T, N, 2)
        chunk_visibility = pred_visibility[0].cpu().numpy()  # (chunk_T, N)
        
        # Save last positions for next chunk
        last_positions = chunk_tracks[-1].copy()
        
        # Handle overlap: skip first `overlap` frames except for first chunk
        if chunk_start == 0:
            all_tracks.append(chunk_tracks)
            all_visibility.append(chunk_visibility)
        else:
            all_tracks.append(chunk_tracks[overlap:])
            all_visibility.append(chunk_visibility[overlap:])
        
        # Move to next chunk (with overlap)
        if chunk_end >= T:
            break
        chunk_start = chunk_end - overlap
        
        # Clear GPU memory
        del video_tensor, pred_tracks, pred_visibility
        torch.cuda.empty_cache()
    
    # Concatenate all chunks
    tracks = np.concatenate(all_tracks, axis=0)[:T]  # (T, N, 2)
    visibility = np.concatenate(all_visibility, axis=0)[:T]  # (T, N)
    
    # Scale tracks back to original resolution
    tracks[:, :, 0] *= scale_x
    tracks[:, :, 1] *= scale_y
    
    track_time = time.time() - start_time
    print(f"Tracking complete in {track_time:.1f}s ({T/track_time:.1f} fps)")
    
    # Visualize and save (using original resolution frames)
    if output_path or show_preview:
        print("Creating visualization...")
        visualize_tracks(frames, tracks, visibility, fps, output_path, show_preview)
    
    print(f"\nOutput shape: {tracks.shape}")
    return tracks, visibility


def visualize_tracks(
    frames: np.ndarray,
    tracks: np.ndarray,
    visibility: np.ndarray,
    fps: float,
    output_path: str = None,
    show_preview: bool = True
):
    """Draw tracking results on video."""
    T, H, W, _ = frames.shape
    N = tracks.shape[1]
    
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
        (128, 255, 0), (0, 128, 255)
    ]
    
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    
    if show_preview:
        cv2.namedWindow("CoTracker Results", cv2.WINDOW_NORMAL)
    
    trail_length = 15
    
    for t in range(T):
        # Convert RGB to BGR for OpenCV
        vis = cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR)
        
        for i in range(N):
            color = colors[i % len(colors)]
            x, y = tracks[t, i]
            is_visible = visibility[t, i] > 0.5
            
            if is_visible:
                # Draw current point
                cv2.circle(vis, (int(x), int(y)), 6, color, -1)
                cv2.circle(vis, (int(x), int(y)), 8, (0, 0, 0), 2)
            else:
                # Draw hollow circle for occluded points
                cv2.circle(vis, (int(x), int(y)), 6, color, 2)
            
            # Draw trail
            trail_start = max(0, t - trail_length)
            for t_trail in range(trail_start, t):
                if visibility[t_trail, i] > 0.5 and visibility[t_trail + 1, i] > 0.5:
                    pt1 = tracks[t_trail, i].astype(int)
                    pt2 = tracks[t_trail + 1, i].astype(int)
                    alpha = (t_trail - trail_start) / trail_length
                    thickness = max(1, int(3 * alpha))
                    cv2.line(vis, tuple(pt1), tuple(pt2), color, thickness)
        
        # Frame counter
        cv2.putText(vis, f"Frame {t}/{T}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if output_path:
            out.write(vis)
        
        if show_preview:
            cv2.imshow("CoTracker Results", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Pause
                cv2.waitKey(0)
    
    if output_path:
        out.release()
        print(f"Saved to: {output_path}")
    
    if show_preview:
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Track points with CoTracker")
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("-o", "--output", help="Output video path")
    parser.add_argument("--no-preview", action="store_true", help="Disable preview")
    parser.add_argument("--save-tracks", help="Save tracks to .npz file")
    parser.add_argument("--chunk-size", type=int, default=60, 
                        help="Frames per chunk (lower = less VRAM, default: 60)")
    parser.add_argument("--resize", type=float, default=0.5,
                        help="Resize factor (0.5 = half resolution, default: 0.5)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Only process first N frames (for testing)")
    
    args = parser.parse_args()
    
    # Default output path
    if args.output is None:
        video_path = Path(args.video)
        args.output = str(video_path.parent / f"{video_path.stem}_cotracker.mp4")
    
    # Run tracking
    tracks, visibility = track_with_cotracker(
        args.video,
        output_path=args.output,
        show_preview=not args.no_preview,
        max_frames_per_chunk=args.chunk_size,
        resize_factor=args.resize,
        max_frames=args.max_frames
    )
    
    # Save tracks
    if args.save_tracks:
        np.savez(args.save_tracks, tracks=tracks, visibility=visibility)
        print(f"Saved tracks to: {args.save_tracks}")
    else:
        tracks_path = Path(args.video).with_suffix('.npz')
        np.savez(tracks_path, tracks=tracks, visibility=visibility)
        print(f"Saved tracks to: {tracks_path}")


if __name__ == "__main__":
    main()