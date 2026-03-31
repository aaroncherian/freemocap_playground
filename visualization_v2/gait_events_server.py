"""
Gait Events Viewer — Python Server
===================================
Loads skeleton data via skellymodels, detects gait events (heel strike / toe off)
using the Zeni velocity-based method, and serves everything to the HTML viewer.

Usage:
    python gait_events_server.py

Edit the paths and settings below, then run. Browser opens automatically.
"""

from __future__ import annotations

import json
import socket
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from skellymodels.managers.human import Human


# ════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit these
# ════════════════════════════════════════════════════════════════════

path_to_recording = Path(
    r"D:\validation\data\2026_01_26_KK\2026-01-16_14-15-39_GMT-5_kk_treadmill_1"
)

# Which tracker to visualize (pick one — this viewer shows a single skeleton)
tracker_id = "mediapipe"

# Where you saved gait_events_viewer.html
path_to_viewer_folder = Path(
    r"C:\Users\aaron\Documents\GitHub\freemocap_playground\visualization_v2"
)

# Recording framerate
FRAMERATE = 30  # Hz — adjust to match your recording

# Color for the skeleton
SKELETON_COLOR = 0x2f6efc  # blue

# Heel landmark names — adjust if your model uses different names
# These need to match what's in your skellymodels anatomical structure
LEFT_HEEL = "left_heel"
RIGHT_HEEL = "right_heel"
LEFT_TOE = "left_foot_index"
RIGHT_TOE = "right_foot_index"

# Sacrum / pelvis marker for AP velocity reference (Zeni method)
SACRUM = "pelvis_center"  # or "mid_hip" or whatever your model calls it

# Gait event detection parameters
VELOCITY_SMOOTHING_WINDOW = 7      # frames — median filter window for smoothing
MIN_FRAMES_BETWEEN_EVENTS = 15     # minimum frames between consecutive events (same side)


# ════════════════════════════════════════════════════════════════════
# Gait Event Detection (Zeni Velocity-Based Method)
# ════════════════════════════════════════════════════════════════════

def smooth(signal: np.ndarray, window: int = 7) -> np.ndarray:
    """Simple moving-average smoothing."""
    if window < 2:
        return signal.copy()
    kernel = np.ones(window) / window
    # Pad to reduce edge effects
    pad = window // 2
    padded = np.pad(signal, pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(signal)]


def detect_gait_events_zeni(
    positions: np.ndarray,
    landmarks: list[str],
    fps: float,
) -> dict:
    """
    Detect heel strikes and toe offs using the Zeni (2008) velocity-based method.

    Heel strike: anterior-posterior (AP) velocity of the heel marker crosses zero
                 from positive to negative (foot decelerating in front of body).
    Toe off:     AP velocity of the toe marker crosses zero from negative to positive
                 (foot accelerating behind the body).

    The AP axis is defined relative to the sacrum (pelvis center) marker.

    Parameters
    ----------
    positions : np.ndarray, shape (F, M, 3)
    landmarks : list of str, length M
    fps : float

    Returns
    -------
    dict with 'heel_strikes' and 'toe_offs', each a list of {frame, side}
    """
    name_to_idx = {name: i for i, name in enumerate(landmarks)}

    def get_idx(name):
        idx = name_to_idx.get(name)
        if idx is None:
            # Try common alternatives
            alternatives = {
                "pelvis_center": ["mid_hip", "hips_center", "sacrum", "pelvis"],
                "left_heel": ["l_heel", "left_heel_marker"],
                "right_heel": ["r_heel", "right_heel_marker"],
                "left_foot_index": ["l_toe", "left_toe", "left_foot_index_marker", "left_big_toe"],
                "right_foot_index": ["r_toe", "right_toe", "right_foot_index_marker", "right_big_toe"],
            }
            for alt in alternatives.get(name, []):
                if alt in name_to_idx:
                    return name_to_idx[alt]
            raise KeyError(
                f"Landmark '{name}' not found. Available: {landmarks[:20]}..."
            )
        return idx

    sacrum_idx = get_idx(SACRUM)
    l_heel_idx = get_idx(LEFT_HEEL)
    r_heel_idx = get_idx(RIGHT_HEEL)
    l_toe_idx = get_idx(LEFT_TOE)
    r_toe_idx = get_idx(RIGHT_TOE)

    F = positions.shape[0]

    # Extract AP (Y-axis) positions relative to sacrum
    sacrum_y = positions[:, sacrum_idx, 1]

    l_heel_y_rel = positions[:, l_heel_idx, 1] - sacrum_y
    r_heel_y_rel = positions[:, r_heel_idx, 1] - sacrum_y
    l_toe_y_rel = positions[:, l_toe_idx, 1] - sacrum_y
    r_toe_y_rel = positions[:, r_toe_idx, 1] - sacrum_y

    # Compute AP velocity (first derivative)
    dt = 1.0 / fps
    l_heel_vy = smooth(np.gradient(l_heel_y_rel, dt), VELOCITY_SMOOTHING_WINDOW)
    r_heel_vy = smooth(np.gradient(r_heel_y_rel, dt), VELOCITY_SMOOTHING_WINDOW)
    l_toe_vy = smooth(np.gradient(l_toe_y_rel, dt), VELOCITY_SMOOTHING_WINDOW)
    r_toe_vy = smooth(np.gradient(r_toe_y_rel, dt), VELOCITY_SMOOTHING_WINDOW)

    def find_zero_crossings(vel, direction="neg_to_pos"):
        """Find frames where velocity crosses zero in the given direction."""
        events = []
        for i in range(1, len(vel)):
            if direction == "pos_to_neg":
                # Heel strike: velocity goes from positive to negative
                if vel[i - 1] > 0 and vel[i] <= 0:
                    events.append(i)
            elif direction == "neg_to_pos":
                # Toe off: velocity goes from negative to positive
                if vel[i - 1] < 0 and vel[i] >= 0:
                    events.append(i)
        return events

    def filter_min_spacing(events, min_frames):
        """Remove events that are too close together."""
        if not events:
            return events
        filtered = [events[0]]
        for e in events[1:]:
            if e - filtered[-1] >= min_frames:
                filtered.append(e)
        return filtered

    # Heel strikes: heel AP velocity crosses from positive to negative
    l_hs = filter_min_spacing(find_zero_crossings(l_heel_vy, "pos_to_neg"), MIN_FRAMES_BETWEEN_EVENTS)
    r_hs = filter_min_spacing(find_zero_crossings(r_heel_vy, "pos_to_neg"), MIN_FRAMES_BETWEEN_EVENTS)

    # Toe offs: toe AP velocity crosses from negative to positive
    l_to = filter_min_spacing(find_zero_crossings(l_toe_vy, "neg_to_pos"), MIN_FRAMES_BETWEEN_EVENTS)
    r_to = filter_min_spacing(find_zero_crossings(r_toe_vy, "neg_to_pos"), MIN_FRAMES_BETWEEN_EVENTS)

    heel_strikes = (
        [{"frame": int(f), "side": "left"} for f in l_hs]
        + [{"frame": int(f), "side": "right"} for f in r_hs]
    )
    heel_strikes.sort(key=lambda e: e["frame"])

    toe_offs = (
        [{"frame": int(f), "side": "left"} for f in l_to]
        + [{"frame": int(f), "side": "right"} for f in r_to]
    )
    toe_offs.sort(key=lambda e: e["frame"])

    print(f"  Detected {len(l_hs)} L heel strikes, {len(r_hs)} R heel strikes")
    print(f"  Detected {len(l_to)} L toe offs, {len(r_to)} R toe offs")

    return {
        "heel_strikes": heel_strikes,
        "toe_offs": toe_offs,
        # Include velocity traces for optional debug plotting
        "debug": {
            "l_heel_vy": l_heel_vy.tolist(),
            "r_heel_vy": r_heel_vy.tolist(),
            "l_toe_vy": l_toe_vy.tolist(),
            "r_toe_vy": r_toe_vy.tolist(),
        },
    }


# ════════════════════════════════════════════════════════════════════
# Data Loading
# ════════════════════════════════════════════════════════════════════

def convert_connections(segment_connections: dict, landmark_names: list[str]):
    valid = set(landmark_names)
    converted = []
    for seg in segment_connections.values():
        a = seg.get("proximal")
        b = seg.get("distal")
        if a in valid and b in valid:
            converted.append([a, b])
    return converted


def load_tracker_data(recording: Path, tid: str) -> dict:
    """Load a single tracker's data and detect gait events."""
    p = recording / "validation" / tid
    print(f"Loading {tid} from {p} ...")

    h = Human.from_data(p)
    landmarks = h.body.anatomical_structure.landmark_names
    segment_connections = h.body.anatomical_structure.segment_connections
    connections = convert_connections(segment_connections, landmarks)

    positions_array = h.body.xyz.as_array  # numpy array (F, M, 3)
    print(f"  Shape: {positions_array.shape}  ({len(landmarks)} landmarks)")

    # Detect gait events
    print("  Detecting gait events ...")
    gait_events = detect_gait_events_zeni(positions_array, landmarks, FRAMERATE)

    # Build landmark name → index map for the JS side
    landmark_indices = {
        "left_heel": landmarks.index(LEFT_HEEL) if LEFT_HEEL in landmarks else None,
        "right_heel": landmarks.index(RIGHT_HEEL) if RIGHT_HEEL in landmarks else None,
        "left_toe": landmarks.index(LEFT_TOE) if LEFT_TOE in landmarks else None,
        "right_toe": landmarks.index(RIGHT_TOE) if RIGHT_TOE in landmarks else None,
    }
    # Try alternatives if primary names not found
    for key, primary in [
        ("left_heel", LEFT_HEEL), ("right_heel", RIGHT_HEEL),
        ("left_toe", LEFT_TOE), ("right_toe", RIGHT_TOE),
    ]:
        if landmark_indices[key] is None:
            for i, name in enumerate(landmarks):
                if key.replace("_", " ") in name.lower().replace("_", " "):
                    landmark_indices[key] = i
                    break

    return {
        "tracker": getattr(h, "tracker", None) or tid,
        "positions": positions_array.tolist(),
        "connections": connections,
        "landmarks": landmarks,
        "gait_events": {
            "heel_strikes": gait_events["heel_strikes"],
            "toe_offs": gait_events["toe_offs"],
        },
        "landmark_indices": landmark_indices,
        "fps": FRAMERATE,
        "color": SKELETON_COLOR,
    }


# ════════════════════════════════════════════════════════════════════
# Server
# ════════════════════════════════════════════════════════════════════

def pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def run_server():
    viewer_dir = path_to_viewer_folder.resolve()
    recording = path_to_recording.resolve()

    if not (viewer_dir / "gait_events_viewer.html").exists():
        raise FileNotFoundError(
            f"Missing gait_events_viewer.html in {viewer_dir}\n"
            f"Copy the HTML file there first!"
        )

    # Load data and detect events
    payload = load_tracker_data(recording, tracker_id)
    payload_bytes = json.dumps(payload).encode("utf-8")

    print(f"\n  Payload size: {len(payload_bytes) / 1024 / 1024:.1f} MB")

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(viewer_dir), **kwargs)

        def do_GET(self):
            path = self.path.split("?", 1)[0]

            # Serve the skeleton + gait events data
            if path == "/gait_data.json":
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload_bytes)))
                self.end_headers()
                self.wfile.write(payload_bytes)
                return

            return super().do_GET()

        def log_message(self, format, *args):
            # Quieter logging — only show non-200 or data requests
            if "gait_data.json" in str(args) or "404" in str(args):
                super().log_message(format, *args)

    port = pick_free_port()
    host = "127.0.0.1"
    url = f"http://{host}:{port}/gait_events_viewer.html"

    print(f"\n{'═' * 60}")
    print(f"  Gait Events Viewer")
    print(f"  Recording:  {recording}")
    print(f"  Tracker:    {tracker_id}")
    print(f"  Framerate:  {FRAMERATE} Hz")
    print(f"  Server:     http://{host}:{port}")
    print(f"  Viewer:     {url}")
    print(f"{'═' * 60}\n")

    server = ThreadingHTTPServer((host, port), Handler)
    threading.Timer(0.3, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    run_server()
