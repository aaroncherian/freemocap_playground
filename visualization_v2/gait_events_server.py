"""
Gait Events Viewer — Python Server
===================================
Loads skeleton data via skellymodels and pre-computed gait events from CSV,
then serves everything to the HTML viewer.

Usage:
    python gait_events_server.py

Edit the paths and settings below, then run. Browser opens automatically.
"""

from __future__ import annotations

import csv
import json
import socket
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from skellymodels.managers.human import Human


# ════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit these
# ════════════════════════════════════════════════════════════════════

path_to_recording = Path(
    r"D:\validation\data\2026_01_26_KK\2026-01-16_14-15-39_GMT-5_kk_treadmill_1"
)

# Which tracker to visualize
tracker_id = "rtmpose"

# Where you saved gait_events_viewer.html
path_to_viewer_folder = Path(
    r"C:\Users\aaron\Documents\GitHub\freemocap_playground\visualization_v2"
)

# Recording framerate
FRAMERATE = 30  # Hz

# Skeleton color (hex int)
SKELETON_COLOR = 0x6b5847  # walnut brown to match viewer palette

# Heel / toe landmark names for the JS viewer to highlight
LEFT_HEEL = "left_heel"
RIGHT_HEEL = "right_heel"
LEFT_TOE = "left_foot_index"
RIGHT_TOE = "right_foot_index"


# ════════════════════════════════════════════════════════════════════
# Gait Event Loading (from CSV)
# ════════════════════════════════════════════════════════════════════

def find_gait_events_csv(tracker_dir: Path) -> Path | None:
    """
    Look for a gait events CSV inside tracker_dir/gait_events/.
    Searches for any file matching *gait_events.csv.
    """
    gait_events_dir = tracker_dir / "gait_events"
    if not gait_events_dir.exists():
        print(f"  Warning: gait_events directory not found at {gait_events_dir}")
        return None

    candidates = list(gait_events_dir.glob("*gait_events.csv"))
    if not candidates:
        print(f"  Warning: no *gait_events.csv found in {gait_events_dir}")
        return None

    # If multiple, prefer the one matching the tracker name
    for c in candidates:
        if tracker_id in c.stem.lower():
            return c
    return candidates[0]


def load_gait_events_csv(csv_path: Path) -> dict:
    """
    Parse gait events CSV with columns: foot, event, frame

    Returns dict with 'heel_strikes' and 'toe_offs',
    each a list of {frame: int, side: str}.
    """
    heel_strikes = []
    toe_offs = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            foot = row["foot"].strip().lower()
            event = row["event"].strip().lower()
            frame = int(row["frame"])

            entry = {"frame": frame, "side": foot}

            if event == "heel_strike":
                heel_strikes.append(entry)
            elif event == "toe_off":
                toe_offs.append(entry)
            else:
                print(f"  Warning: unknown event type '{event}' at frame {frame}")

    heel_strikes.sort(key=lambda e: e["frame"])
    toe_offs.sort(key=lambda e: e["frame"])

    l_hs = sum(1 for e in heel_strikes if e["side"] == "left")
    r_hs = sum(1 for e in heel_strikes if e["side"] == "right")
    l_to = sum(1 for e in toe_offs if e["side"] == "left")
    r_to = sum(1 for e in toe_offs if e["side"] == "right")

    print(f"  Loaded {len(heel_strikes)} heel strikes ({l_hs} L, {r_hs} R)")
    print(f"  Loaded {len(toe_offs)} toe offs ({l_to} L, {r_to} R)")

    return {
        "heel_strikes": heel_strikes,
        "toe_offs": toe_offs,
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


def find_landmark_index(landmarks: list[str], target: str) -> int | None:
    """Find index of target landmark name, trying alternatives if needed."""
    if target in landmarks:
        return landmarks.index(target)

    # Try fuzzy match
    target_normalized = target.lower().replace("_", " ")
    for i, name in enumerate(landmarks):
        if target_normalized in name.lower().replace("_", " "):
            return i
    return None


def load_tracker_data(recording: Path, tid: str) -> dict:
    """Load skeleton data + gait events for a single tracker."""
    tracker_dir = recording / "validation" / tid
    print(f"Loading {tid} from {tracker_dir} ...")

    # Load skeleton
    h = Human.from_data(tracker_dir)
    landmarks = h.body.anatomical_structure.landmark_names
    segment_connections = h.body.anatomical_structure.segment_connections
    connections = convert_connections(segment_connections, landmarks)

    positions_array = h.body.xyz.as_array  # numpy (F, M, 3)
    print(f"  Shape: {positions_array.shape}  ({len(landmarks)} landmarks)")

    # Load gait events from CSV
    csv_path = find_gait_events_csv(tracker_dir)
    if csv_path:
        print(f"  Loading gait events from {csv_path.name} ...")
        gait_events = load_gait_events_csv(csv_path)
    else:
        print("  No gait events CSV found — viewer will show skeleton only")
        gait_events = {"heel_strikes": [], "toe_offs": []}

    # Build landmark indices for the JS viewer
    landmark_indices = {
        "left_heel": find_landmark_index(landmarks, LEFT_HEEL),
        "right_heel": find_landmark_index(landmarks, RIGHT_HEEL),
        "left_toe": find_landmark_index(landmarks, LEFT_TOE),
        "right_toe": find_landmark_index(landmarks, RIGHT_TOE),
    }

    missing = [k for k, v in landmark_indices.items() if v is None]
    if missing:
        print(f"  Warning: could not find landmarks: {missing}")
        print(f"  Available: {landmarks[:15]}...")

    return {
        "tracker": getattr(h, "tracker", None) or tid,
        "positions": positions_array.tolist(),
        "connections": connections,
        "landmarks": landmarks,
        "gait_events": gait_events,
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

    # Load data
    payload = load_tracker_data(recording, tracker_id)
    payload_bytes = json.dumps(payload).encode("utf-8")

    n_hs = len(payload["gait_events"]["heel_strikes"])
    n_to = len(payload["gait_events"]["toe_offs"])
    n_frames = len(payload["positions"])

    print(f"\n  Payload size: {len(payload_bytes) / 1024 / 1024:.1f} MB")

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(viewer_dir), **kwargs)

        def do_GET(self):
            path = self.path.split("?", 1)[0]

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
            if "gait_data.json" in str(args) or "404" in str(args):
                super().log_message(format, *args)

    port = pick_free_port()
    host = "127.0.0.1"
    url = f"http://{host}:{port}/gait_events_viewer.html"

    print(f"\n{'═' * 60}")
    print(f"  Gait Events Viewer")
    print(f"  Recording:  {recording}")
    print(f"  Tracker:    {tracker_id}")
    print(f"  Frames:     {n_frames} @ {FRAMERATE} Hz")
    print(f"  Events:     {n_hs} heel strikes, {n_to} toe offs")
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