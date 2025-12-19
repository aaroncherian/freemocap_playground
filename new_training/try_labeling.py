"""
Point Labeling GUI

Label points across video frames for training data. Supports manual labeling
and optical flow assisted propagation with anchor frames.

Configure your marker names in the MARKER_NAMES list at the top of this file.

Visibility states:
    0 = Unlabeled (no data for this point on this frame)
    1 = Occluded (labeled but not visible - your best guess)
    2 = Visible (labeled and visible)

Visual guide:
    Solid circle     = visible (state 2)
    Dashed circle    = occluded (state 1)
    No marker        = unlabeled (state 0)
    Yellow highlight = selected point
    Cyan bar         = optical flow range
    Orange banner    = anchor labeling mode

Controls:
    1-9         Select marker by number
    Left click  Place selected marker (visible)
    Right click Toggle visible/occluded
    Backspace   Clear marker on current frame
    
    a/d or Left/Right   Navigate frames (1 at a time)
    w/s or Up/Down      Jump 10 frames
    Trackbar            Scrub to any frame
    
    [           Set optical flow start
    ]           Set optical flow end
    i           Set anchor interval
    o           Run optical flow
    r           Reset OF range
    
    q           Confirm anchor / Quit (context dependent)
    Escape      Exit anchor mode
    
    z               Undo
    x               Save/export labels
    l               Load labels
    h               Help

Usage:
    python labeler.py path/to/video.mp4
"""

import argparse
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List
import cv2
import numpy as np


# =============================================================================
# CONFIGURATION - Edit these marker names for your project
# =============================================================================
MARKER_NAMES = [
    "left_board",
    "front_board",
    "mid_board",
    "roller"
]
# =============================================================================


# Visibility states
UNLABELED = 0
OCCLUDED = 1
VISIBLE = 2


@dataclass
class Action:
    """For undo support."""
    frame_idx: int
    point_idx: int
    old_pos: Optional[np.ndarray]
    old_vis: int
    new_pos: Optional[np.ndarray]
    new_vis: int


class PointLabeler:
    def __init__(self, video_path: str, marker_names: List[str] = None):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Marker names from config
        self.marker_names = marker_names if marker_names else MARKER_NAMES
        self.n_points = len(self.marker_names)
        
        # Initialize label arrays
        self.points = np.full((self.n_frames, self.n_points, 2), np.nan)
        self.visibility = np.zeros((self.n_frames, self.n_points), dtype=np.uint8)
        
        # UI state
        self.current_frame_idx = 0
        self.selected_point = 0
        self.frame_cache = {}
        self.undo_stack = []
        self.max_undo = 50
        
        # Anchor labeling mode
        self.anchor_queue = []  # Frames waiting to be labeled
        self.in_anchor_mode = False
        
        # UI state
        self.current_frame_idx = 0
        self.selected_point = 0
        self.frame_cache = {}
        self.undo_stack = []
        self.max_undo = 50
        
        # Point colors
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
            (128, 255, 0), (0, 128, 255)
        ]
        
        # LK params for optical flow
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        # Optical flow range
        self.of_start = None
        self.of_end = None
        self.anchor_interval = 30
        
        print(f"Loaded: {video_path}")
        print(f"  {self.n_frames} frames, {self.width}x{self.height}, {self.fps:.1f} fps")
        print(f"\nMarkers ({self.n_points}):")
        for i, name in enumerate(self.marker_names):
            print(f"  {i+1}: {name}")
        print(f"\nPress 'h' for help")
    
    def get_frame(self, idx: int) -> np.ndarray:
        if idx in self.frame_cache:
            return self.frame_cache[idx].copy()
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        
        if ret and len(self.frame_cache) < 100:
            self.frame_cache[idx] = frame.copy()
        
        return frame if ret else None
    
    def add_point(self):
        """Add a new point track."""
        self.n_points += 1
        
        if self.points is None:
            self.points = np.full((self.n_frames, 1, 2), np.nan)
            self.visibility = np.zeros((self.n_frames, 1), dtype=np.uint8)
        else:
            new_pts = np.full((self.n_frames, 1, 2), np.nan)
            new_vis = np.zeros((self.n_frames, 1), dtype=np.uint8)
            self.points = np.concatenate([self.points, new_pts], axis=1)
            self.visibility = np.concatenate([self.visibility, new_vis], axis=1)
        
        self.selected_point = self.n_points - 1
        print(f"Added point {self.n_points}. Click to place it.")
    
    def delete_point_track(self, point_idx: int):
        """Delete an entire point track."""
        if self.n_points <= 0:
            return
        
        self.points = np.delete(self.points, point_idx, axis=1)
        self.visibility = np.delete(self.visibility, point_idx, axis=1)
        self.n_points -= 1
        
        if self.n_points == 0:
            self.points = None
            self.visibility = None
            self.selected_point = 0
        else:
            self.selected_point = min(self.selected_point, self.n_points - 1)
        
        print(f"Deleted point track. {self.n_points} points remaining.")
    
    def set_point(self, frame_idx: int, point_idx: int, x: float, y: float, vis: int):
        """Set point position and visibility with undo support."""
        if self.points is None or point_idx >= self.n_points:
            return
        
        # Record for undo
        old_pos = self.points[frame_idx, point_idx].copy()
        old_vis = self.visibility[frame_idx, point_idx]
        
        action = Action(frame_idx, point_idx, 
                       old_pos if not np.isnan(old_pos).any() else None, old_vis,
                       np.array([x, y]), vis)
        self.undo_stack.append(action)
        if len(self.undo_stack) > self.max_undo:
            self.undo_stack.pop(0)
        
        # Apply
        self.points[frame_idx, point_idx] = [x, y]
        self.visibility[frame_idx, point_idx] = vis
    
    def clear_point(self, frame_idx: int, point_idx: int):
        """Clear a point on a specific frame."""
        if self.points is None or point_idx >= self.n_points:
            return
        
        old_pos = self.points[frame_idx, point_idx].copy()
        old_vis = self.visibility[frame_idx, point_idx]
        
        action = Action(frame_idx, point_idx,
                       old_pos if not np.isnan(old_pos).any() else None, old_vis,
                       None, UNLABELED)
        self.undo_stack.append(action)
        
        self.points[frame_idx, point_idx] = [np.nan, np.nan]
        self.visibility[frame_idx, point_idx] = UNLABELED
    
    def toggle_visibility(self, frame_idx: int, point_idx: int):
        """Toggle between visible and occluded."""
        if self.points is None or point_idx >= self.n_points:
            return
        
        if self.visibility[frame_idx, point_idx] == UNLABELED:
            return  # Can't toggle unlabeled
        
        old_vis = self.visibility[frame_idx, point_idx]
        new_vis = OCCLUDED if old_vis == VISIBLE else VISIBLE
        
        action = Action(frame_idx, point_idx,
                       self.points[frame_idx, point_idx].copy(), old_vis,
                       self.points[frame_idx, point_idx].copy(), new_vis)
        self.undo_stack.append(action)
        
        self.visibility[frame_idx, point_idx] = new_vis
    
    def undo(self):
        """Undo last action."""
        if not self.undo_stack:
            print("Nothing to undo")
            return
        
        action = self.undo_stack.pop()
        
        if action.old_pos is None:
            self.points[action.frame_idx, action.point_idx] = [np.nan, np.nan]
        else:
            self.points[action.frame_idx, action.point_idx] = action.old_pos
        
        self.visibility[action.frame_idx, action.point_idx] = action.old_vis
        print(f"Undid action on frame {action.frame_idx}, point {action.point_idx + 1}")
    
    def find_nearest_point(self, x: int, y: int, max_dist: float = 30) -> Optional[int]:
        """Find the nearest labeled point to a click location."""
        if self.points is None:
            return None
        
        pts = self.points[self.current_frame_idx]
        min_dist = float('inf')
        nearest = None
        
        for i in range(self.n_points):
            if self.visibility[self.current_frame_idx, i] == UNLABELED:
                continue
            dist = np.sqrt((pts[i, 0] - x)**2 + (pts[i, 1] - y)**2)
            if dist < min_dist and dist < max_dist:
                min_dist = dist
                nearest = i
        
        return nearest
    
    def track_frame_pair(self, frame1, frame2, points):
        """Track points between two frames with forward-backward check."""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        pts = points.reshape(-1, 1, 2).astype(np.float32)
        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, pts, None, **self.lk_params)
        back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(gray2, gray1, new_pts, None, **self.lk_params)
        
        fb_error = np.linalg.norm(pts - back_pts, axis=2).flatten()
        confidence = np.exp(-fb_error / 1.5)
        confidence[(status.flatten() != 1) | (back_status.flatten() != 1)] = 0
        
        return new_pts.reshape(-1, 2), confidence
    
    def run_optical_flow_with_anchors(self):
        """Run optical flow between anchors with bidirectional blending."""
        if self.of_start is None or self.of_end is None:
            print("Set start ([) and end (]) frames first")
            return
        
        start, end = self.of_start, self.of_end
        
        # Determine anchor frames
        anchor_indices = list(range(start, end + 1, self.anchor_interval))
        if end not in anchor_indices:
            anchor_indices.append(end)
        
        print(f"\nOptical flow range: {start} -> {end}")
        print(f"Anchor interval: {self.anchor_interval}")
        print(f"Anchors at frames: {anchor_indices}")
        
        # Check which anchors need labeling
        anchors_to_label = []
        for idx in anchor_indices:
            if not (self.visibility[idx] > 0).all():
                anchors_to_label.append(idx)
        
        if anchors_to_label:
            print(f"\nNeed labels at anchors: {anchors_to_label}")
            print("Label all points on each anchor, press 'q' to confirm and go to next.")
            print("Press Escape to exit anchor mode.\n")
            
            # Enter anchor labeling mode
            self.anchor_queue = anchors_to_label.copy()
            self.in_anchor_mode = True
            
            # Jump to first unlabeled anchor
            self.current_frame_idx = self.anchor_queue[0]
            try:
                cv2.setTrackbarPos("Frame", "Labeler", self.current_frame_idx)
            except:
                pass
            return
        
        print("\nAll anchors labeled. Running bidirectional tracking...")
        
        # Cache frames for speed
        print("Caching frames...")
        frames = {}
        for i in range(start, end + 1):
            frames[i] = self.get_frame(i)
        
        # Get anchor positions
        anchor_points = {idx: self.points[idx].copy() for idx in anchor_indices}
        anchor_vis = {idx: self.visibility[idx].copy() for idx in anchor_indices}
        
        # Track between each pair of anchors
        sorted_anchors = sorted(anchor_indices)
        
        for a_idx in range(len(sorted_anchors) - 1):
            start_anchor = sorted_anchors[a_idx]
            end_anchor = sorted_anchors[a_idx + 1]
            seg_len = end_anchor - start_anchor
            
            print(f"  Tracking {start_anchor} -> {end_anchor}...")
            
            # Forward pass
            fwd_tracks = np.zeros((seg_len + 1, self.n_points, 2))
            fwd_conf = np.zeros((seg_len + 1, self.n_points))
            fwd_tracks[0] = anchor_points[start_anchor]
            fwd_conf[0] = 1.0
            
            for j in range(seg_len):
                fwd_tracks[j+1], conf = self.track_frame_pair(
                    frames[start_anchor + j],
                    frames[start_anchor + j + 1],
                    fwd_tracks[j]
                )
                fwd_conf[j+1] = conf * (fwd_conf[j] ** 0.98)
            
            # Backward pass
            bwd_tracks = np.zeros((seg_len + 1, self.n_points, 2))
            bwd_conf = np.zeros((seg_len + 1, self.n_points))
            bwd_tracks[-1] = anchor_points[end_anchor]
            bwd_conf[-1] = 1.0
            
            for j in range(seg_len, 0, -1):
                bwd_tracks[j-1], conf = self.track_frame_pair(
                    frames[start_anchor + j],
                    frames[start_anchor + j - 1],
                    bwd_tracks[j]
                )
                bwd_conf[j-1] = conf * (bwd_conf[j] ** 0.98)
            
            # Blend and store (skip anchors, they're already labeled)
            for j in range(1, seg_len):
                frame_idx = start_anchor + j
                
                # Skip frames that are already fully labeled
                if (self.visibility[frame_idx] > 0).all():
                    continue
                
                t = j / seg_len  # Blend factor
                
                for p in range(self.n_points):
                    # Skip if already labeled
                    if self.visibility[frame_idx, p] > 0:
                        continue
                    
                    w_fwd = fwd_conf[j, p] * (1 - t)
                    w_bwd = bwd_conf[j, p] * t
                    total = w_fwd + w_bwd + 1e-8
                    
                    blended = (w_fwd * fwd_tracks[j, p] + w_bwd * bwd_tracks[j, p]) / total
                    
                    # Inherit visibility from nearest anchor
                    if t < 0.5:
                        vis = anchor_vis[start_anchor][p]
                    else:
                        vis = anchor_vis[end_anchor][p]
                    
                    self.set_point(frame_idx, p, blended[0], blended[1], vis)
        
        n_filled = ((self.visibility[start:end+1] > 0).sum() - 
                    len(anchor_indices) * self.n_points)
        print(f"\nDone! Filled {n_filled} point-frames via optical flow.")
        print("Review results and refine as needed.")
    
    def draw_frame(self, frame: np.ndarray) -> np.ndarray:
        """Draw labels and UI on frame."""
        display = frame.copy()
        h, w = display.shape[:2]
        
        # Draw points
        if self.points is not None:
            for i in range(self.n_points):
                vis = self.visibility[self.current_frame_idx, i]
                if vis == UNLABELED:
                    continue
                
                pt = self.points[self.current_frame_idx, i]
                if np.isnan(pt).any():
                    continue
                
                x, y = int(pt[0]), int(pt[1])
                color = self.colors[i % len(self.colors)]
                is_selected = (i == self.selected_point)
                
                if vis == VISIBLE:
                    # Solid circle
                    cv2.circle(display, (x, y), 8, color, -1)
                    cv2.circle(display, (x, y), 8, (255, 255, 255), 1)
                else:  # OCCLUDED
                    # Dashed circle effect (draw partial arcs)
                    for angle in range(0, 360, 30):
                        start = angle * np.pi / 180
                        end = (angle + 15) * np.pi / 180
                        pts = []
                        for a in np.linspace(start, end, 5):
                            pts.append([x + 8*np.cos(a), y + 8*np.sin(a)])
                        pts = np.array(pts, dtype=np.int32)
                        cv2.polylines(display, [pts], False, color, 2)
                    cv2.circle(display, (x, y), 3, color, -1)
                
                # Selection highlight
                if is_selected:
                    cv2.circle(display, (x, y), 14, (0, 255, 255), 2)
                
                # Point number/name
                label = f"{i+1}"
                cv2.putText(display, label, (x + 12, y + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Bottom bar
        overlay = display.copy()
        cv2.rectangle(overlay, (0, h - 50), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
        
        # Frame info
        cv2.putText(display, f"Frame: {self.current_frame_idx}/{self.n_frames-1}",
                   (10, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Point info
        if self.n_points > 0:
            marker_name = self.marker_names[self.selected_point]
            pt_info = f"[{self.selected_point + 1}] {marker_name}"
            cv2.putText(display, pt_info, (10, h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Labeled count for this frame
        if self.visibility is not None:
            n_labeled = (self.visibility[self.current_frame_idx] > 0).sum()
            cv2.putText(display, f"Labeled: {n_labeled}/{self.n_points}",
                       (w - 140, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # OF range info
        if self.of_start is not None and self.of_end is not None:
            of_info = f"OF: [{self.of_start}-{self.of_end}] int:{self.anchor_interval}"
            cv2.putText(display, of_info, (w - 200, h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        elif self.of_start is not None:
            cv2.putText(display, f"OF: [{self.of_start}-?]", (w - 140, h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # Timeline
        timeline_y = h - 45
        timeline_start, timeline_end = 50, w - 50
        timeline_width = timeline_end - timeline_start
        
        cv2.line(display, (timeline_start, timeline_y), (timeline_end, timeline_y), (100, 100, 100), 2)
        
        # Show OF range
        if self.of_start is not None:
            sx = int(timeline_start + (self.of_start / self.n_frames) * timeline_width)
            cv2.line(display, (sx, timeline_y - 8), (sx, timeline_y + 8), (0, 255, 255), 2)
        if self.of_end is not None:
            ex = int(timeline_start + (self.of_end / self.n_frames) * timeline_width)
            cv2.line(display, (ex, timeline_y - 8), (ex, timeline_y + 8), (0, 255, 255), 2)
        if self.of_start is not None and self.of_end is not None:
            sx = int(timeline_start + (self.of_start / self.n_frames) * timeline_width)
            ex = int(timeline_start + (self.of_end / self.n_frames) * timeline_width)
            cv2.rectangle(display, (sx, timeline_y - 3), (ex, timeline_y + 3), (0, 255, 255), -1)
        
        # Show labeled frames on timeline
        if self.visibility is not None:
            for f in range(self.n_frames):
                if (self.visibility[f] > 0).any():
                    fx = int(timeline_start + (f / self.n_frames) * timeline_width)
                    cv2.circle(display, (fx, timeline_y), 2, (0, 255, 0), -1)
        
        # Current position
        curr_x = int(timeline_start + (self.current_frame_idx / self.n_frames) * timeline_width)
        cv2.circle(display, (curr_x, timeline_y), 5, (255, 255, 255), -1)
        
        # Anchor mode indicator
        if self.in_anchor_mode:
            cv2.rectangle(display, (0, 0), (w, 30), (0, 100, 200), -1)
            remaining = len(self.anchor_queue)
            cv2.putText(display, f"ANCHOR MODE - Label all points, press 'q' to confirm ({remaining} anchors remaining)",
                       (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return display
    
    def save_labels(self, path: Optional[str] = None):
        """Save labels to file."""
        if self.points is None:
            print("No labels to save")
            return
        
        if path is None:
            path = Path(self.video_path).with_suffix('.labels.npz')
        
        # Count stats
        total = self.n_frames * self.n_points
        n_visible = (self.visibility == VISIBLE).sum()
        n_occluded = (self.visibility == OCCLUDED).sum()
        n_unlabeled = (self.visibility == UNLABELED).sum()
        
        np.savez(
            path,
            points=self.points,
            visibility=self.visibility,
            marker_names=np.array(self.marker_names),
            n_points=self.n_points,
            n_frames=self.n_frames,
            video_path=str(self.video_path),
            fps=self.fps
        )
        
        print(f"\nSaved to: {path}")
        print(f"  {self.n_points} points × {self.n_frames} frames")
        print(f"  Visible: {n_visible} ({100*n_visible/total:.1f}%)")
        print(f"  Occluded: {n_occluded} ({100*n_occluded/total:.1f}%)")
        print(f"  Unlabeled: {n_unlabeled} ({100*n_unlabeled/total:.1f}%)")
    
    def load_labels(self, path: Optional[str] = None):
        """Load labels from file."""
        if path is None:
            path = Path(self.video_path).with_suffix('.labels.npz')
        
        if not Path(path).exists():
            print(f"No labels file found: {path}")
            return
        
        data = np.load(path, allow_pickle=True)
        self.points = data['points']
        self.visibility = data['visibility']
        self.n_points = int(data['n_points'])
        
        if 'marker_names' in data:
            self.marker_names = list(data['marker_names'])
        
        print(f"Loaded {self.n_points} point tracks from {path}")
    
    def show_help(self):
        """Print help."""
        print("""
=== POINT LABELING GUI ===

POINT SELECTION:
    1-9             Select marker by number

LABELING:
    Left click      Place selected marker (visible)
    Right click     Toggle visible/occluded
    Backspace       Clear marker on this frame

NAVIGATION:
    Left/Right or a/d    Previous/next frame
    Up/Down or w/s       Jump 10 frames
    Trackbar             Scrub to any frame

OPTICAL FLOW:
    [               Set OF start frame
    ]               Set OF end frame
    i               Set anchor interval
    o               Run optical flow
    r               Reset OF range
    
    (In anchor mode)
    q               Confirm frame, go to next anchor
    Escape          Exit anchor mode

EDIT:
    z               Undo

FILE:
    x               Save/export labels
    l               Load labels
    
OTHER:
    h               This help
    q               Quit (when not in anchor mode)

VISUAL GUIDE:
    Solid circle    = Visible (state 2)
    Dashed circle   = Occluded (state 1)
    Yellow ring     = Selected marker
    Cyan bar        = Optical flow range
    Orange bar      = Anchor mode active
        """)
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Place the selected point
            self.set_point(self.current_frame_idx, self.selected_point, 
                          float(x), float(y), VISIBLE)
            
            # Auto-advance to next unlabeled point on this frame
            for i in range(1, self.n_points):
                next_idx = (self.selected_point + i) % self.n_points
                if self.visibility[self.current_frame_idx, next_idx] == UNLABELED:
                    self.selected_point = next_idx
                    print(f"  -> [{next_idx + 1}] {self.marker_names[next_idx]}")
                    break
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Toggle visibility of nearest point
            nearest = self.find_nearest_point(x, y)
            if nearest is not None:
                self.toggle_visibility(self.current_frame_idx, nearest)
                vis_name = "occluded" if self.visibility[self.current_frame_idx, nearest] == OCCLUDED else "visible"
                print(f"{self.marker_names[nearest]} -> {vis_name}")
    
    def run(self):
        """Main loop."""
        cv2.namedWindow("Labeler", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Labeler", self.mouse_callback)
        cv2.createTrackbar("Frame", "Labeler", 0, self.n_frames - 1, lambda x: None)
        
        # Try to load existing labels
        self.load_labels()
        
        while True:
            # Sync with trackbar
            trackbar_pos = cv2.getTrackbarPos("Frame", "Labeler")
            if trackbar_pos != self.current_frame_idx:
                self.current_frame_idx = trackbar_pos
            
            frame = self.get_frame(self.current_frame_idx)
            if frame is None:
                break
            
            display = self.draw_frame(frame)
            cv2.imshow("Labeler", display)
            
            key = cv2.waitKey(30) & 0xFF
            
            # Anchor mode: q confirms and moves to next
            if self.in_anchor_mode:
                if key == ord('q'):
                    # Check if current frame is fully labeled
                    if (self.visibility[self.current_frame_idx] > 0).all():
                        # Remove from queue
                        if self.current_frame_idx in self.anchor_queue:
                            self.anchor_queue.remove(self.current_frame_idx)
                        
                        if self.anchor_queue:
                            # Go to next anchor
                            self.current_frame_idx = self.anchor_queue[0]
                            cv2.setTrackbarPos("Frame", "Labeler", self.current_frame_idx)
                            print(f"Next anchor: frame {self.current_frame_idx} ({len(self.anchor_queue)} remaining)")
                        else:
                            # All done - run the tracking
                            print("\nAll anchors labeled! Running optical flow...")
                            self.in_anchor_mode = False
                            self.run_optical_flow_with_anchors()
                    else:
                        n_missing = (self.visibility[self.current_frame_idx] == 0).sum()
                        print(f"Label all points first ({n_missing} missing)")
                    continue
                elif key == 27:  # Escape
                    print("Exited anchor mode")
                    self.in_anchor_mode = False
                    self.anchor_queue = []
                    continue
            
            # Quit
            if key == ord('q') and not self.in_anchor_mode:
                break
            
            # Help
            elif key == ord('h'):
                self.show_help()
            
            # Point selection (1-9)
            elif ord('1') <= key <= ord('9'):
                idx = key - ord('1')
                if idx < self.n_points:
                    self.selected_point = idx
                    print(f"Selected: [{idx + 1}] {self.marker_names[idx]}")
            
            # Clear point on frame
            elif key == 8 or key == 127:  # Backspace
                if self.n_points > 0:
                    self.clear_point(self.current_frame_idx, self.selected_point)
                    print(f"Cleared point {self.selected_point + 1} on frame {self.current_frame_idx}")
            
            # Navigation - arrow keys vary by platform, so also support a/d and w/s
            # Left arrow or 'a' - back 1 frame
            elif key in [81, 63234, ord('a')]:
                self.current_frame_idx = max(0, self.current_frame_idx - 1)
                cv2.setTrackbarPos("Frame", "Labeler", self.current_frame_idx)
            # Right arrow or 'd' - forward 1 frame
            elif key in [83, 63235, ord('d')]:
                self.current_frame_idx = min(self.n_frames - 1, self.current_frame_idx + 1)
                cv2.setTrackbarPos("Frame", "Labeler", self.current_frame_idx)
            # Up arrow or 'w' - jump 10 forward
            elif key in [82, 63232, ord('w')]:
                self.current_frame_idx = min(self.n_frames - 1, self.current_frame_idx + 10)
                cv2.setTrackbarPos("Frame", "Labeler", self.current_frame_idx)
            # Down arrow or 's' - jump 10 back
            elif key in [84, 63233, ord('s')]:
                self.current_frame_idx = max(0, self.current_frame_idx - 10)
                cv2.setTrackbarPos("Frame", "Labeler", self.current_frame_idx)
            
            # Optical flow
            elif key == ord('['):
                self.of_start = self.current_frame_idx
                print(f"OF start: {self.of_start}")
            elif key == ord(']'):
                if self.of_start is not None and self.current_frame_idx > self.of_start:
                    self.of_end = self.current_frame_idx
                    print(f"OF end: {self.of_end}")
                else:
                    print("Set start ([) first, and end must be after start")
            elif key == ord('i'):
                try:
                    val = input(f"Anchor interval ({self.anchor_interval}): ").strip()
                    if val:
                        self.anchor_interval = int(val)
                        print(f"Anchor interval: {self.anchor_interval}")
                except ValueError:
                    print("Invalid interval")
            elif key == ord('o'):
                self.run_optical_flow_with_anchors()
            elif key == ord('r'):
                self.of_start = None
                self.of_end = None
                print("Reset OF range")
            
            # Undo - 'z'
            elif key == ord('z'):
                self.undo()
            
            # Save - 'x' for export
            elif key == ord('x'):
                self.save_labels()
            
            # Load - 'l'
            elif key == ord('l'):
                self.load_labels()
        
        cv2.destroyAllWindows()
        self.cap.release()

def main(video_path: str):

    labeler = PointLabeler(video_path)
    labeler.run()


if __name__ == "__main__":
    main(r"D:\sfn\michael_wobble\recording_12_07_09_gmt-5__MDN_wobble_3\synchronized_videos\Camera_000_synchronized.mp4")