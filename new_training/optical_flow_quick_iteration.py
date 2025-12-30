"""
Interactive Optical Flow Explorer

Quick feedback loop for testing optical flow tracking.

Visual guide:
    - Solid circle = high confidence (≥0.6)
    - Orange ring = medium confidence (0.3-0.6)
    - Red X = low confidence (<threshold)
    - White ring = anchor frame (your labels)

Controls:
    s/e         Set start/end frame
    r           Reset range
    a           Set anchor interval
    t           Set confidence threshold
    l           Label anchors & run tracking
    p           Playback results
    x           Export tracks
    h           Help
    q           Quit

Usage:
    python explorer.py path/to/video.mp4
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def draw_dashed_line(img, pt1, pt2, color, thickness=1, dash_length=10):
    """Draw a dashed line between two points."""
    pt1 = np.array(pt1, dtype=float)
    pt2 = np.array(pt2, dtype=float)
    
    dist = np.linalg.norm(pt2 - pt1)
    if dist < 1:
        return
    
    direction = (pt2 - pt1) / dist
    n_dashes = int(dist / dash_length)
    
    for i in range(0, n_dashes, 2):
        start = pt1 + direction * i * dash_length
        end = pt1 + direction * min((i + 1) * dash_length, dist)
        cv2.line(img, tuple(start.astype(int)), tuple(end.astype(int)), color, thickness)


class OpticalFlowExplorer:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # State
        self.current_frame_idx = 0
        self.start_frame = None
        self.end_frame = None
        self.anchor_interval = 10
        
        # Tracking results
        self.tracks = None
        self.confidence = None
        self.anchor_indices = []
        
        # Cache
        self.cached_frames = None
        self.cache_start = None
        
        # LK params
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.anchor_interval , 0.01)
        )
        
        # Confidence threshold for warnings
        self.confidence_threshold = 0.3
        
        print(f"Loaded: {video_path}")
        print(f"  {self.n_frames} frames, {self.width}x{self.height}, {self.fps:.1f} fps")
        print(f"\nPress 'h' for help")
    
    def get_frame(self, idx: int) -> np.ndarray:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def draw_ui(self, frame: np.ndarray) -> np.ndarray:
        display = frame.copy()
        h, w = display.shape[:2]
        
        # Bottom bar
        overlay = display.copy()
        cv2.rectangle(overlay, (0, h - 60), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
        
        # Frame info
        cv2.putText(display, f"Frame: {self.current_frame_idx}/{self.n_frames-1}", 
                   (10, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Range info
        if self.start_frame is not None and self.end_frame is not None:
            range_info = f"Range: [{self.start_frame} - {self.end_frame}]"
            color = (0, 255, 0)
        elif self.start_frame is not None:
            range_info = f"Range: [{self.start_frame} - ???]"
            color = (0, 255, 255)
        else:
            range_info = "No range (press 's')"
            color = (128, 128, 128)
        
        cv2.putText(display, range_info, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        
        # Anchor interval
        cv2.putText(display, f"Anchor: {self.anchor_interval}", 
                   (w - 150, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if self.tracks is not None:
            cv2.putText(display, f"Tracked: {self.tracks.shape[1]} pts",
                       (w - 150, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        # Timeline
        timeline_y = h - 55
        timeline_start = 50
        timeline_end = w - 50
        timeline_width = timeline_end - timeline_start
        
        cv2.line(display, (timeline_start, timeline_y), (timeline_end, timeline_y), (100, 100, 100), 2)
        
        curr_x = int(timeline_start + (self.current_frame_idx / self.n_frames) * timeline_width)
        cv2.circle(display, (curr_x, timeline_y), 5, (255, 255, 255), -1)
        
        if self.start_frame is not None:
            start_x = int(timeline_start + (self.start_frame / self.n_frames) * timeline_width)
            cv2.line(display, (start_x, timeline_y - 8), (start_x, timeline_y + 8), (0, 255, 0), 2)
        
        if self.end_frame is not None:
            end_x = int(timeline_start + (self.end_frame / self.n_frames) * timeline_width)
            cv2.line(display, (end_x, timeline_y - 8), (end_x, timeline_y + 8), (0, 0, 255), 2)
        
        if self.start_frame is not None and self.end_frame is not None:
            start_x = int(timeline_start + (self.start_frame / self.n_frames) * timeline_width)
            end_x = int(timeline_start + (self.end_frame / self.n_frames) * timeline_width)
            cv2.rectangle(display, (start_x, timeline_y - 3), (end_x, timeline_y + 3), (0, 255, 0), -1)
        
        return display
    
    def draw_tracks_on_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """Draw tracking results with confidence visualization."""
        if self.tracks is None:
            return frame
        
        display = frame.copy()
        relative_idx = frame_idx - self.cache_start
        
        if relative_idx < 0 or relative_idx >= len(self.tracks):
            return display
        
        colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0),
                  (255,0,255), (0,255,255), (128,0,255), (255,128,0)]
        
        n_points = self.tracks.shape[1]
        is_anchor = frame_idx in self.anchor_indices
        
        n_low_conf = 0
        
        for i in range(n_points):
            x, y = self.tracks[relative_idx, i]
            base_color = colors[i % len(colors)]
            conf = self.confidence[relative_idx, i] if self.confidence is not None else 1.0
            
            cx, cy = int(x), int(y)
            
            if is_anchor:
                # Anchor: white ring + solid dot
                cv2.circle(display, (cx, cy), 10, (255, 255, 255), 2)
                cv2.circle(display, (cx, cy), 5, base_color, -1)
            
            elif conf < self.confidence_threshold:
                # LOW CONFIDENCE - red X warning
                n_low_conf += 1
                cv2.circle(display, (cx, cy), 12, (0, 0, 255), 2)
                cv2.line(display, (cx - 6, cy - 6), (cx + 6, cy + 6), (0, 0, 255), 2)
                cv2.line(display, (cx - 6, cy + 6), (cx + 6, cy - 6), (0, 0, 255), 2)
                dim_color = tuple(int(c * 0.4) for c in base_color)
                cv2.circle(display, (cx, cy), 3, dim_color, -1)
                cv2.putText(display, f"{conf:.2f}", (cx + 10, cy - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            
            elif conf < 0.6:
                # MEDIUM CONFIDENCE - orange ring
                cv2.circle(display, (cx, cy), 8, (0, 165, 255), 2)
                color_adj = tuple(int(c * 0.7) for c in base_color)
                cv2.circle(display, (cx, cy), 4, color_adj, -1)
                
            else:
                # HIGH CONFIDENCE - solid circle
                cv2.circle(display, (cx, cy), 5, base_color, -1)
            
            # Trail
            trail_len = min(15, relative_idx)
            for t in range(relative_idx - trail_len, relative_idx):
                if t >= 0:
                    pt1 = tuple(self.tracks[t, i].astype(int))
                    pt2 = tuple(self.tracks[t + 1, i].astype(int))
                    seg_conf = min(self.confidence[t, i], self.confidence[t + 1, i])
                    
                    if seg_conf < self.confidence_threshold:
                        draw_dashed_line(display, pt1, pt2, (0, 0, 255), thickness=1, dash_length=4)
                    elif seg_conf < 0.6:
                        cv2.line(display, pt1, pt2, (0, 165, 255), 1)
                    else:
                        cv2.line(display, pt1, pt2, base_color, 2)
        
        # Warning banner
        if n_low_conf > 0:
            cv2.rectangle(display, (0, 0), (250, 35), (0, 0, 180), -1)
            cv2.putText(display, f"LOW CONFIDENCE: {n_low_conf} pts", 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return display
    
    def select_points(self, frame: np.ndarray, reference_pts=None) -> np.ndarray:
        points = []
        display = frame.copy()
        
        if reference_pts is not None:
            for i, (x, y) in enumerate(reference_pts):
                cv2.circle(display, (int(x), int(y)), 10, (0, 255, 255), 2)
                cv2.putText(display, str(i+1), (int(x)+12, int(y)+5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        base_display = display.copy()
        
        def click(event, x, y, flags, param):
            nonlocal display
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append([x, y])
                cv2.circle(display, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(display, str(len(points)), (x+8, y+5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imshow("Label Points", display)
        
        cv2.namedWindow("Label Points", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Label Points", click)
        cv2.imshow("Label Points", display)
        
        print("Click points. 'q'=done, 'r'=reset, 's'=skip (use reference)")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') and len(points) > 0:
                break
            elif key == ord('s') and reference_pts is not None:
                cv2.destroyWindow("Label Points")
                return reference_pts.copy()
            elif key == ord('r'):
                points.clear()
                display = base_display.copy()
                cv2.imshow("Label Points", display)
        
        cv2.destroyWindow("Label Points")
        return np.array(points, dtype=np.float32)
    
    def cache_range(self):
        if self.start_frame is None or self.end_frame is None:
            return False
        
        print(f"Loading frames {self.start_frame}-{self.end_frame}...")
        self.cached_frames = []
        self.cache_start = self.start_frame
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        for i in range(self.end_frame - self.start_frame + 1):
            ret, frame = self.cap.read()
            if not ret:
                break
            self.cached_frames.append(frame)
        
        print(f"Cached {len(self.cached_frames)} frames")
        return True
    
    def track_frame_pair(self, prev_frame, curr_frame, prev_points):
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        prev_pts = prev_points.reshape(-1, 1, 2).astype(np.float32)
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **self.lk_params)
        back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(curr_gray, prev_gray, curr_pts, None, **self.lk_params)
        
        fb_error = np.linalg.norm(prev_pts - back_pts, axis=2).flatten()
        confidence = np.exp(-fb_error / 1.5)
        confidence[(status.flatten() != 1) | (back_status.flatten() != 1)] = 0
        
        return curr_pts.reshape(-1, 2), confidence
    
    def run_tracking(self):
        if self.cached_frames is None:
            print("No frames cached. Set range and press 'l'")
            return
        
        n_frames = len(self.cached_frames)
        
        # Anchor frames
        self.anchor_indices = list(range(self.start_frame, self.end_frame + 1, self.anchor_interval))
        if self.end_frame not in self.anchor_indices:
            self.anchor_indices.append(self.end_frame)
        
        print(f"Anchors at frames: {self.anchor_indices}")
        
        # Label anchors
        anchor_points = {}
        for i, frame_idx in enumerate(self.anchor_indices):
            relative_idx = frame_idx - self.start_frame
            frame = self.cached_frames[relative_idx]
            
            print(f"\nLabel anchor {i+1}/{len(self.anchor_indices)} (frame {frame_idx})")
            ref_pts = anchor_points.get(self.anchor_indices[i-1]) if i > 0 else None
            points = self.select_points(frame, reference_pts=ref_pts)
            anchor_points[frame_idx] = points
        
        n_points = len(anchor_points[self.anchor_indices[0]])
        
        # Init output
        self.tracks = np.zeros((n_frames, n_points, 2), dtype=np.float32)
        self.confidence = np.zeros((n_frames, n_points), dtype=np.float32)
        
        # Fill anchors
        for idx in self.anchor_indices:
            rel_idx = idx - self.start_frame
            self.tracks[rel_idx] = anchor_points[idx]
            self.confidence[rel_idx] = 1.0
        
        # Track between anchors (bidirectional)
        sorted_anchors = sorted(self.anchor_indices)
        
        for i in range(len(sorted_anchors) - 1):
            start_anchor = sorted_anchors[i]
            end_anchor = sorted_anchors[i + 1]
            
            start_rel = start_anchor - self.start_frame
            end_rel = end_anchor - self.start_frame
            seg_len = end_rel - start_rel
            
            print(f"Tracking {start_anchor} -> {end_anchor}...")
            
            # Forward
            fwd = np.zeros((seg_len + 1, n_points, 2))
            fwd_conf = np.zeros((seg_len + 1, n_points))
            fwd[0] = anchor_points[start_anchor]
            fwd_conf[0] = 1.0
            
            for j in range(seg_len):
                fwd[j+1], fwd_conf[j+1] = self.track_frame_pair(
                    self.cached_frames[start_rel + j],
                    self.cached_frames[start_rel + j + 1],
                    fwd[j]
                )
                fwd_conf[j+1] *= fwd_conf[j] ** 0.98
            
            # Backward
            bwd = np.zeros((seg_len + 1, n_points, 2))
            bwd_conf = np.zeros((seg_len + 1, n_points))
            bwd[-1] = anchor_points[end_anchor]
            bwd_conf[-1] = 1.0
            
            for j in range(seg_len, 0, -1):
                bwd[j-1], bwd_conf[j-1] = self.track_frame_pair(
                    self.cached_frames[start_rel + j],
                    self.cached_frames[start_rel + j - 1],
                    bwd[j]
                )
                bwd_conf[j-1] *= bwd_conf[j] ** 0.98
            
            # Blend
            for j in range(1, seg_len):
                t = j / seg_len
                w_fwd = fwd_conf[j] * (1 - t)
                w_bwd = bwd_conf[j] * t
                total = w_fwd + w_bwd + 1e-8
                
                rel_idx = start_rel + j
                self.tracks[rel_idx] = (w_fwd[:, None] * fwd[j] + w_bwd[:, None] * bwd[j]) / total[:, None]
                self.confidence[rel_idx] = (w_fwd + w_bwd) / 2
        
        print(f"\nTracking complete! Press 'p' to playback")
    
    def playback(self):
        if self.tracks is None or self.cached_frames is None:
            print("No tracking results. Run tracking first.")
            return
        
        print("Playback: space=pause, q=stop, </>=step")
        cv2.namedWindow("Playback", cv2.WINDOW_NORMAL)
        
        paused = False
        idx = 0
        
        while True:
            frame = self.cached_frames[idx].copy()
            frame = self.draw_tracks_on_frame(frame, self.start_frame + idx)
            
            info_text = f"Frame {self.start_frame + idx}"
            if self.start_frame + idx in self.anchor_indices:
                info_text += " [ANCHOR]"
            
            cv2.putText(frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            mean_conf = self.confidence[idx].mean()
            conf_color = (0, 255, 0) if mean_conf > 0.6 else (0, 165, 255) if mean_conf > 0.3 else (0, 0, 255)
            cv2.putText(frame, f"Avg conf: {mean_conf:.2f}", (10, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, conf_color, 1)
            
            cv2.imshow("Playback", frame)
            
            key = cv2.waitKey(30 if not paused else 0) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
            elif key == ord(',') or key == ord('<') or key == 81:
                idx = max(0, idx - 1)
                paused = True
            elif key == ord('.') or key == ord('>') or key == 83:
                idx = min(len(self.cached_frames) - 1, idx + 1)
                paused = True
            
            if not paused:
                idx = (idx + 1) % len(self.cached_frames)
        
        cv2.destroyWindow("Playback")
    
    def export_tracks(self):
        if self.tracks is None:
            print("No tracking results to export")
            return
        
        output_path = Path(self.video_path).with_suffix('.tracking.npz')
        
        frame_indices = np.arange(self.start_frame, self.start_frame + len(self.tracks))
        low_confidence_mask = self.confidence < self.confidence_threshold
        
        np.savez(
            output_path,
            tracks=self.tracks,
            confidence=self.confidence,
            low_confidence_mask=low_confidence_mask,
            frame_indices=frame_indices,
            anchor_indices=np.array(self.anchor_indices)
        )
        
        n_frames = len(self.tracks)
        n_points = self.tracks.shape[1]
        n_high = (self.confidence >= self.confidence_threshold).sum()
        n_low = low_confidence_mask.sum()
        
        print(f"\nExported to: {output_path}")
        print(f"  High confidence: {n_high} ({100*n_high/(n_frames*n_points):.1f}%)")
        print(f"  Low confidence: {n_low} ({100*n_low/(n_frames*n_points):.1f}%)")
    
    def show_help(self):
        print("""
        === CONTROLS ===
        s           Set start frame
        e           Set end frame
        r           Reset range
        a           Set anchor interval
        t           Set confidence threshold
        l           Label & track
        p           Playback
        x           Export
        h           Help
        q           Quit
        
        === VISUAL ===
        Solid circle    = high confidence
        Orange ring     = medium confidence
        Red X           = low confidence
        White ring      = anchor (your label)
        """)
    
    def run(self):
        cv2.namedWindow("Explorer", cv2.WINDOW_NORMAL)
        cv2.createTrackbar("Frame", "Explorer", 0, self.n_frames - 1, lambda x: None)
        
        while True:
            trackbar_pos = cv2.getTrackbarPos("Frame", "Explorer")
            if trackbar_pos != self.current_frame_idx:
                self.current_frame_idx = trackbar_pos
            
            frame = self.get_frame(self.current_frame_idx)
            if frame is None:
                break
            
            if self.tracks is not None and self.cache_start is not None:
                if self.start_frame <= self.current_frame_idx <= self.end_frame:
                    frame = self.draw_tracks_on_frame(frame, self.current_frame_idx)
            
            display = self.draw_ui(frame)
            cv2.imshow("Explorer", display)
            
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('h'):
                self.show_help()
            elif key == ord('s'):
                self.start_frame = self.current_frame_idx
                self.tracks = None
                print(f"Start: {self.start_frame}")
            elif key == ord('e'):
                if self.start_frame is not None and self.current_frame_idx > self.start_frame:
                    self.end_frame = self.current_frame_idx
                    self.tracks = None
                    print(f"End: {self.end_frame}")
            elif key == ord('r'):
                self.start_frame = None
                self.end_frame = None
                self.tracks = None
                self.cached_frames = None
                print("Reset")
            elif key == ord('a'):
                try:
                    val = input("Anchor interval: ").strip()
                    self.anchor_interval = int(val)
                    print(f"Anchor interval: {self.anchor_interval}")
                except ValueError:
                    print("Invalid")
            elif key == ord('t'):
                try:
                    val = input(f"Confidence threshold ({self.confidence_threshold}): ").strip()
                    self.confidence_threshold = float(val)
                    print(f"Threshold: {self.confidence_threshold}")
                except ValueError:
                    print("Invalid")
            elif key == ord('l'):
                if self.start_frame is not None and self.end_frame is not None:
                    if self.cache_range():
                        self.run_tracking()
                else:
                    print("Set start/end first")
            elif key == ord('p'):
                self.playback()
            elif key == ord('x'):
                self.export_tracks()
            elif key == 81 or key == 2:  # left
                self.current_frame_idx = max(0, self.current_frame_idx - 1)
                cv2.setTrackbarPos("Frame", "Explorer", self.current_frame_idx)
            elif key == 83 or key == 3:  # right
                self.current_frame_idx = min(self.n_frames - 1, self.current_frame_idx + 1)
                cv2.setTrackbarPos("Frame", "Explorer", self.current_frame_idx)
        
        cv2.destroyAllWindows()
        self.cap.release()


def main(path_to_video:str=None):

    explorer = OpticalFlowExplorer(path_to_video)
    explorer.run()


if __name__ == "__main__":
    main(r"C:\Users\aaron\Downloads\Copy of 1 day 0.3 D21 - DTA M2 R uni Gq - CT2.mov")