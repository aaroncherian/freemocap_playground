"""
Robust Optical Flow Tracker

Improvements over basic optical flow:
- Bidirectional tracking from anchors (forward + backward, blended)
- Confidence estimation per point per frame
- Auto-detection of tracking failures
- Interactive correction of bad frames
- Kalman filtering for smoothness

Usage:
    python robust_tracker.py path/to/video.mp4

Requirements:
    pip install opencv-python numpy scipy
"""

import argparse
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d


@dataclass
class TrackingConfig:
    """Tracking parameters."""
    # Lucas-Kanade params
    lk_win_size: int = 21
    lk_max_level: int = 3
    
    # Forward-backward consistency threshold (pixels)
    fb_threshold: float = 1.5
    
    # Confidence thresholds
    low_confidence_threshold: float = 0.5
    critical_confidence_threshold: float = 0.3
    
    # Occlusion detection
    occlusion_confidence_threshold: float = 0.25  # Below this = occluded
    occlusion_min_frames: int = 2  # Must be low-conf for N frames to count as occlusion
    
    # Motion model for occlusion interpolation
    use_velocity_prediction: bool = True
    velocity_smoothing_frames: int = 5
    
    # Smoothing
    temporal_smooth_sigma: float = 1.0
    
    # Auto-anchor detection
    auto_anchor_confidence_threshold: float = 0.4
    min_frames_between_auto_anchors: int = 20


def select_points(frame: np.ndarray, existing_points: np.ndarray = None,
                  window_name: str = "Click points") -> np.ndarray:
    """Interactive point selection with reference points shown."""
    points = []
    display = frame.copy()
    
    if existing_points is not None:
        for i, (x, y) in enumerate(existing_points):
            cv2.circle(display, (int(x), int(y)), 12, (0, 255, 255), 2)
            cv2.putText(display, str(i+1), (int(x)+14, int(y)+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    base_display = display.copy()
    
    def click_handler(event, x, y, flags, param):
        nonlocal display
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            cv2.circle(display, (x, y), 6, (0, 255, 0), -1)
            cv2.circle(display, (x, y), 8, (0, 0, 0), 2)
            cv2.putText(display, str(len(points)), (x+10, y+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow(window_name, display)
            print(f"  Point {len(points)}: ({x}, {y})")
    
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(1280, frame.shape[1]), min(720, frame.shape[0]))
    cv2.setMouseCallback(window_name, click_handler)
    cv2.imshow(window_name, display)
    
    hint = "Yellow=reference. " if existing_points is not None else ""
    print(f"{hint}Click points, 'q'=done, 'r'=reset, 's'=skip (use tracked)")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') and len(points) > 0:
            break
        elif key == ord('s') and existing_points is not None:
            cv2.destroyAllWindows()
            return existing_points.copy()
        elif key == ord('r'):
            points.clear()
            display = base_display.copy()
            cv2.imshow(window_name, display)
            print("  Reset")
    
    cv2.destroyAllWindows()
    return np.array(points, dtype=np.float32)


class RobustTracker:
    """Robust point tracker with bidirectional tracking and confidence estimation."""
    
    def __init__(self, config: TrackingConfig = None):
        self.config = config or TrackingConfig()
        self.lk_params = dict(
            winSize=(self.config.lk_win_size, self.config.lk_win_size),
            maxLevel=self.config.lk_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
    
    def track_frame_pair(self, prev_frame: np.ndarray, curr_frame: np.ndarray,
                         prev_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Track points with forward-backward consistency check.
        
        Returns:
            points: (N, 2) new positions
            confidence: (N,) confidence scores 0-1
        """
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        prev_pts = prev_points.reshape(-1, 1, 2).astype(np.float32)
        
        # Forward track
        curr_pts, status_fwd, err_fwd = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None, **self.lk_params
        )
        
        # Backward track
        back_pts, status_bwd, err_bwd = cv2.calcOpticalFlowPyrLK(
            curr_gray, prev_gray, curr_pts, None, **self.lk_params
        )
        
        # Forward-backward error
        fb_error = np.linalg.norm(prev_pts - back_pts, axis=2).flatten()
        
        # Compute confidence based on multiple factors
        status_ok = (status_fwd.flatten() == 1) & (status_bwd.flatten() == 1)
        
        # Normalize errors to confidence (lower error = higher confidence)
        fb_conf = np.exp(-fb_error / self.config.fb_threshold)
        fb_conf[~status_ok] = 0
        
        # Error-based confidence (OpenCV returns error as match quality)
        err_conf = np.exp(-err_fwd.flatten() / 50)  # normalize by typical error
        
        # Combined confidence
        confidence = fb_conf * err_conf
        confidence = np.clip(confidence, 0, 1)
        
        return curr_pts.reshape(-1, 2), confidence
    
    def track_bidirectional(self, frames: list, anchor_indices: list, 
                            anchor_points: dict) -> tuple[np.ndarray, np.ndarray]:
        """
        Track using bidirectional interpolation between anchors.
        
        For frames between anchor A and anchor B:
        - Track forward from A
        - Track backward from B
        - Blend based on distance to nearest anchor
        """
        n_frames = len(frames)
        n_points = len(anchor_points[anchor_indices[0]])
        
        tracks = np.zeros((n_frames, n_points, 2), dtype=np.float32)
        confidence = np.zeros((n_frames, n_points), dtype=np.float32)
        
        # Fill anchor frames
        for idx in anchor_indices:
            tracks[idx] = anchor_points[idx]
            confidence[idx] = 1.0
        
        # Process each segment between anchors
        sorted_anchors = sorted(anchor_indices)
        
        for i in range(len(sorted_anchors)):
            start_anchor = sorted_anchors[i]
            end_anchor = sorted_anchors[i + 1] if i + 1 < len(sorted_anchors) else n_frames - 1
            
            segment_len = end_anchor - start_anchor
            if segment_len <= 1:
                continue
            
            print(f"  Tracking segment {start_anchor} -> {end_anchor}")
            
            # Forward tracking from start_anchor
            fwd_tracks = np.zeros((segment_len + 1, n_points, 2))
            fwd_conf = np.zeros((segment_len + 1, n_points))
            fwd_tracks[0] = anchor_points[start_anchor]
            fwd_conf[0] = 1.0
            
            for j in range(segment_len):
                frame_idx = start_anchor + j
                fwd_tracks[j + 1], fwd_conf[j + 1] = self.track_frame_pair(
                    frames[frame_idx], frames[frame_idx + 1], fwd_tracks[j]
                )
                # Decay confidence over distance
                fwd_conf[j + 1] *= fwd_conf[j] ** 0.98
            
            # Backward tracking from end_anchor (if it exists)
            if end_anchor in anchor_points:
                bwd_tracks = np.zeros((segment_len + 1, n_points, 2))
                bwd_conf = np.zeros((segment_len + 1, n_points))
                bwd_tracks[-1] = anchor_points[end_anchor]
                bwd_conf[-1] = 1.0
                
                for j in range(segment_len, 0, -1):
                    frame_idx = start_anchor + j
                    bwd_tracks[j - 1], bwd_conf[j - 1] = self.track_frame_pair(
                        frames[frame_idx], frames[frame_idx - 1], bwd_tracks[j]
                    )
                    bwd_conf[j - 1] *= bwd_conf[j] ** 0.98
                
                # Blend forward and backward based on position in segment
                for j in range(1, segment_len):
                    frame_idx = start_anchor + j
                    
                    # Weight: higher when closer to the respective anchor
                    t = j / segment_len  # 0 at start, 1 at end
                    
                    # Use confidence-weighted blending
                    w_fwd = fwd_conf[j] * (1 - t)
                    w_bwd = bwd_conf[j] * t
                    
                    total_w = w_fwd + w_bwd + 1e-8
                    
                    tracks[frame_idx] = (w_fwd[:, None] * fwd_tracks[j] + 
                                         w_bwd[:, None] * bwd_tracks[j]) / total_w[:, None]
                    confidence[frame_idx] = (w_fwd + w_bwd) / 2
            else:
                # No end anchor - just use forward tracking
                for j in range(1, segment_len + 1):
                    frame_idx = start_anchor + j
                    if frame_idx < n_frames:
                        tracks[frame_idx] = fwd_tracks[j]
                        confidence[frame_idx] = fwd_conf[j]
        
        return tracks, confidence
    
    def smooth_tracks(self, tracks: np.ndarray, confidence: np.ndarray,
                      anchor_indices: list) -> np.ndarray:
        """Apply temporal smoothing while preserving anchor positions."""
        smoothed = tracks.copy()
        
        # Create anchor mask
        anchor_mask = np.zeros(len(tracks), dtype=bool)
        anchor_mask[anchor_indices] = True
        
        # Smooth each point's trajectory
        for i in range(tracks.shape[1]):
            for dim in range(2):
                signal = tracks[:, i, dim].copy()
                
                # Weight by confidence for smoothing
                weights = confidence[:, i].copy()
                weights[anchor_mask] = 10  # Heavy weight on anchors
                
                # Gaussian smooth
                smoothed_signal = gaussian_filter1d(
                    signal * weights, 
                    sigma=self.config.temporal_smooth_sigma
                )
                smoothed_weights = gaussian_filter1d(
                    weights, 
                    sigma=self.config.temporal_smooth_sigma
                )
                
                smoothed[:, i, dim] = smoothed_signal / (smoothed_weights + 1e-8)
        
        # Restore exact anchor positions
        for idx in anchor_indices:
            smoothed[idx] = tracks[idx]
        
        return smoothed
    
    def find_low_confidence_frames(self, confidence: np.ndarray, 
                                   anchor_indices: list) -> list:
        """Find frames where tracking confidence is low (candidates for new anchors)."""
        mean_conf = confidence.mean(axis=1)
        low_conf_frames = []
        
        last_anchor = -self.config.min_frames_between_auto_anchors
        
        for i, conf in enumerate(mean_conf):
            if i in anchor_indices:
                last_anchor = i
                continue
            
            if conf < self.config.auto_anchor_confidence_threshold:
                if i - last_anchor >= self.config.min_frames_between_auto_anchors:
                    low_conf_frames.append(i)
                    last_anchor = i
        
        return low_conf_frames

    def detect_occlusions(self, confidence: np.ndarray) -> np.ndarray:
        """
        Detect occlusion periods for each point.
        
        Returns:
            occlusion_mask: (n_frames, n_points) bool array, True = occluded
        """
        n_frames, n_points = confidence.shape
        occlusion_mask = np.zeros((n_frames, n_points), dtype=bool)
        
        for p in range(n_points):
            # Find runs of low confidence
            low_conf = confidence[:, p] < self.config.occlusion_confidence_threshold
            
            # Only mark as occluded if low for multiple consecutive frames
            run_length = 0
            run_start = 0
            
            for i in range(n_frames):
                if low_conf[i]:
                    if run_length == 0:
                        run_start = i
                    run_length += 1
                else:
                    if run_length >= self.config.occlusion_min_frames:
                        occlusion_mask[run_start:i, p] = True
                    run_length = 0
            
            # Handle run at end
            if run_length >= self.config.occlusion_min_frames:
                occlusion_mask[run_start:n_frames, p] = True
        
        return occlusion_mask

    def estimate_velocity(self, tracks: np.ndarray, confidence: np.ndarray,
                         frame_idx: int, point_idx: int) -> np.ndarray:
        """Estimate velocity for a point based on recent motion."""
        n_frames = self.config.velocity_smoothing_frames
        start = max(0, frame_idx - n_frames)
        
        velocities = []
        for i in range(start, frame_idx):
            if confidence[i, point_idx] > 0.5 and confidence[i + 1, point_idx] > 0.5:
                vel = tracks[i + 1, point_idx] - tracks[i, point_idx]
                velocities.append(vel)
        
        if velocities:
            return np.mean(velocities, axis=0)
        return np.array([0.0, 0.0])

    def interpolate_occlusions(self, tracks: np.ndarray, confidence: np.ndarray,
                               occlusion_mask: np.ndarray, 
                               anchor_indices: list) -> np.ndarray:
        """
        Fill in occluded regions using interpolation and velocity prediction.
        
        Strategy:
        1. If we have visible frames on both sides: spline interpolation
        2. If only visible before: velocity-based prediction
        3. If only visible after: backward velocity prediction
        """
        tracks_fixed = tracks.copy()
        n_frames, n_points = tracks.shape[:2]
        
        for p in range(n_points):
            # Find occlusion segments
            occ = occlusion_mask[:, p]
            if not occ.any():
                continue
            
            # Find start/end of each occlusion segment
            segments = []
            in_segment = False
            seg_start = 0
            
            for i in range(n_frames):
                if occ[i] and not in_segment:
                    seg_start = i
                    in_segment = True
                elif not occ[i] and in_segment:
                    segments.append((seg_start, i))
                    in_segment = False
            
            if in_segment:
                segments.append((seg_start, n_frames))
            
            # Interpolate each segment
            for seg_start, seg_end in segments:
                # Find last good frame before occlusion
                pre_idx = seg_start - 1 if seg_start > 0 else None
                pre_valid = pre_idx is not None and confidence[pre_idx, p] > 0.3
                
                # Find first good frame after occlusion
                post_idx = seg_end if seg_end < n_frames else None
                post_valid = post_idx is not None and confidence[post_idx, p] > 0.3
                
                if pre_valid and post_valid:
                    # Interpolate between known positions
                    pre_pos = tracks[pre_idx, p]
                    post_pos = tracks[post_idx, p]
                    
                    for i in range(seg_start, seg_end):
                        t = (i - pre_idx) / (post_idx - pre_idx)
                        # Smooth interpolation (ease in/out)
                        t_smooth = t * t * (3 - 2 * t)
                        tracks_fixed[i, p] = pre_pos + t_smooth * (post_pos - pre_pos)
                
                elif pre_valid and self.config.use_velocity_prediction:
                    # Predict forward using velocity
                    velocity = self.estimate_velocity(tracks, confidence, pre_idx, p)
                    for i in range(seg_start, seg_end):
                        dt = i - pre_idx
                        # Dampen velocity over time
                        damping = 0.95 ** dt
                        tracks_fixed[i, p] = tracks[pre_idx, p] + velocity * dt * damping
                
                elif post_valid and self.config.use_velocity_prediction:
                    # Predict backward using velocity
                    velocity = self.estimate_velocity(tracks, confidence, post_idx, p)
                    for i in range(seg_end - 1, seg_start - 1, -1):
                        dt = post_idx - i
                        damping = 0.95 ** dt
                        tracks_fixed[i, p] = tracks[post_idx, p] - velocity * dt * damping
        
        return tracks_fixed


def run_robust_tracker(
    video_path: str,
    output_path: str = None,
    anchor_interval: int = 50,
    max_frames: int = 1000,
    num_anchors: int = 10,
    interactive_correction: bool = True
):
    """Main tracking function with robust tracking."""
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    max_frames = min(max_frames, total_frames)
    print(f"Video: {total_frames} frames, {width}x{height}, {fps:.1f} fps")
    print(f"Processing first {max_frames} frames\n")
    
    # Load all frames into memory (needed for bidirectional tracking)
    print("Loading frames...")
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if (i + 1) % 200 == 0:
            print(f"  Loaded {i + 1}/{max_frames} frames")
    print(f"Loaded {len(frames)} frames\n")
    
    # Initial anchor frames
    anchor_indices = [i * anchor_interval for i in range(num_anchors) 
                      if i * anchor_interval < len(frames)]
    
    # Label initial anchors
    anchor_points = {}
    for i, idx in enumerate(anchor_indices):
        print(f"\n=== Label anchor frame {idx} ({i+1}/{len(anchor_indices)}) ===")
        prev_pts = anchor_points.get(anchor_indices[i-1]) if i > 0 else None
        points = select_points(frames[idx], existing_points=prev_pts,
                              window_name=f"Frame {idx}")
        anchor_points[idx] = points
    
    # Initialize tracker
    tracker = RobustTracker()
    
    # Track with current anchors
    print("\n=== Running bidirectional tracking ===")
    tracks, confidence = tracker.track_bidirectional(frames, anchor_indices, anchor_points)
    
    # Find low confidence frames
    if interactive_correction:
        low_conf_frames = tracker.find_low_confidence_frames(confidence, anchor_indices)
        
        if low_conf_frames:
            print(f"\n=== Found {len(low_conf_frames)} low-confidence frames ===")
            print(f"Frames: {low_conf_frames[:10]}{'...' if len(low_conf_frames) > 10 else ''}")
            
            response = input("\nAdd corrections? (y/n/some): ").strip().lower()
            
            if response == 'y':
                frames_to_correct = low_conf_frames
            elif response == 'some':
                # Take evenly spaced subset
                n = min(5, len(low_conf_frames))
                step = len(low_conf_frames) // n
                frames_to_correct = low_conf_frames[::step][:n]
            else:
                frames_to_correct = []
            
            for idx in frames_to_correct:
                print(f"\n=== Correct frame {idx} ===")
                # Show current tracked position as reference
                points = select_points(frames[idx], existing_points=tracks[idx],
                                      window_name=f"Correct frame {idx}")
                anchor_points[idx] = points
                anchor_indices.append(idx)
            
            if frames_to_correct:
                # Re-track with new anchors
                anchor_indices = sorted(anchor_indices)
                print("\n=== Re-running tracking with corrections ===")
                tracks, confidence = tracker.track_bidirectional(frames, anchor_indices, anchor_points)
    
    # Smooth tracks
    print("\nSmoothing tracks...")
    tracks_smoothed = tracker.smooth_tracks(tracks, confidence, anchor_indices)
    
    # Visualize
    print("\nCreating visualization...")
    visualize_with_confidence(
        frames, tracks_smoothed, confidence, anchor_indices,
        fps, output_path
    )
    
    # Save
    tracks_path = Path(video_path).with_suffix('.npz')
    np.savez(tracks_path,
             tracks=tracks_smoothed,
             tracks_raw=tracks,
             confidence=confidence,
             anchor_indices=np.array(anchor_indices))
    print(f"Saved to: {tracks_path}")
    
    cap.release()
    return tracks_smoothed, confidence


def visualize_with_confidence(frames, tracks, confidence, anchor_indices,
                               fps, output_path=None):
    """Visualize tracks with confidence coloring."""
    n_frames = len(frames)
    n_points = tracks.shape[1]
    height, width = frames[0].shape[:2]
    
    # Color palette
    point_colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), 
                    (255,0,255), (0,255,255), (128,0,255), (255,128,0)]
    
    if output_path:
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    cv2.namedWindow("Robust Tracking", cv2.WINDOW_NORMAL)
    
    for t in range(n_frames):
        frame = frames[t].copy()
        is_anchor = t in anchor_indices
        mean_conf = confidence[t].mean()
        
        # Draw confidence bar
        bar_width = int(mean_conf * 200)
        bar_color = (0, int(255 * mean_conf), int(255 * (1 - mean_conf)))
        cv2.rectangle(frame, (10, height - 30), (10 + bar_width, height - 10), bar_color, -1)
        cv2.rectangle(frame, (10, height - 30), (210, height - 10), (255, 255, 255), 1)
        
        for i in range(n_points):
            x, y = tracks[t, i]
            base_color = point_colors[i % len(point_colors)]
            conf = confidence[t, i]
            
            # Modulate color by confidence
            color = tuple(int(c * (0.3 + 0.7 * conf)) for c in base_color)
            
            if is_anchor:
                cv2.circle(frame, (int(x), int(y)), 12, (255, 255, 255), 2)
            
            radius = 6 if conf > 0.5 else 4
            cv2.circle(frame, (int(x), int(y)), radius, color, -1)
            
            # Trail
            for t2 in range(max(0, t - 15), t):
                pt1 = tuple(tracks[t2, i].astype(int))
                pt2 = tuple(tracks[t2 + 1, i].astype(int))
                trail_conf = confidence[t2:t2+2, i].mean()
                trail_color = tuple(int(c * trail_conf) for c in base_color)
                cv2.line(frame, pt1, pt2, trail_color, 2)
        
        status = "ANCHOR" if is_anchor else f"conf={mean_conf:.2f}"
        cv2.putText(frame, f"Frame {t}/{n_frames} {status}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if output_path:
            out.write(frame)
        
        cv2.imshow("Robust Tracking", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)
    
    if output_path:
        out.release()
        print(f"Saved to: {output_path}")
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Robust optical flow tracker")
    parser.add_argument("video", help="Input video")
    parser.add_argument("-o", "--output", help="Output video path")
    parser.add_argument("--anchor-interval", type=int, default=50)
    parser.add_argument("--max-frames", type=int, default=1000)
    parser.add_argument("--num-anchors", type=int, default=10)
    parser.add_argument("--no-interactive", action="store_true",
                        help="Skip interactive correction prompts")
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = str(Path(args.video).parent / f"{Path(args.video).stem}_robust.mp4")
    
    run_robust_tracker(
        args.video,
        output_path=args.output,
        anchor_interval=args.anchor_interval,
        max_frames=args.max_frames,
        num_anchors=args.num_anchors,
        interactive_correction=not args.no_interactive
    )


if __name__ == "__main__":
    main()