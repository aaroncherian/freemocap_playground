"""
Gait Visualization Server
==========================
Loads skeleton data + available data layers (gait events, joint angles, etc.)
and serves everything to the HTML viewer.

Data is auto-discovered from the recording folder structure:
  recording/validation/{tracker}/
  ├── (parquet)                        → skeleton via Human.from_data()
  ├── gait_events/*gait_events.csv     → gait events layer
  └── joint_angles/*joint_angles.csv   → joint angles layer

Usage:
    python gait_events_server.py
"""

from __future__ import annotations

import csv
import json
import socket
import threading
import webbrowser
from collections import defaultdict
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from skellymodels.managers.human import Human


# ════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════

path_to_recording = Path(
    r"D:\validation\data\2026_01_26_KK\2026-01-16_14-15-39_GMT-5_kk_treadmill_1"
)

tracker_id = "mediapipe"

path_to_viewer_folder = Path(
    r"C:\Users\aaron\Documents\GitHub\freemocap_playground\visualization_v2"
)

FRAMERATE = 30
SKELETON_COLOR = 0x6b5847

LEFT_HEEL = "left_heel"
RIGHT_HEEL = "right_heel"
LEFT_TOE = "left_foot_index"
RIGHT_TOE = "right_foot_index"


# ════════════════════════════════════════════════════════════════════
# Layer: Gait Events
# ════════════════════════════════════════════════════════════════════

def find_file(directory: Path, glob_pattern: str, prefer_substring: str = "") -> Path | None:
    if not directory.exists():
        return None
    candidates = list(directory.glob(glob_pattern))
    if not candidates:
        return None
    for c in candidates:
        if prefer_substring and prefer_substring in c.stem.lower():
            return c
    return candidates[0]


def load_gait_events(tracker_dir: Path) -> dict | None:
    csv_path = find_file(tracker_dir / "gait_events", "*gait_events.csv", tracker_id)
    if not csv_path:
        return None

    print(f"  Loading gait events from {csv_path.name} ...")
    heel_strikes, toe_offs = [], []

    with open(csv_path, "r", newline="") as f:
        for row in csv.DictReader(f):
            entry = {"frame": int(row["frame"]), "side": row["foot"].strip().lower()}
            event = row["event"].strip().lower()
            if event == "heel_strike":
                heel_strikes.append(entry)
            elif event == "toe_off":
                toe_offs.append(entry)

    heel_strikes.sort(key=lambda e: e["frame"])
    toe_offs.sort(key=lambda e: e["frame"])
    print(f"    {len(heel_strikes)} heel strikes, {len(toe_offs)} toe offs")
    return {"heel_strikes": heel_strikes, "toe_offs": toe_offs}


# ════════════════════════════════════════════════════════════════════
# Layer: Joint Angles
# ════════════════════════════════════════════════════════════════════

def load_joint_angles(tracker_dir: Path) -> dict | None:
    """
    Load joint angles CSV (columns: frame, side, joint, angle, component).

    Returns a nested dict structure optimized for the JS viewer:
    {
      "joints": ["knee", "hip", "ankle"],
      "components": ["flex_ext", "abd_add", ...],
      "sides": ["left", "right"],
      "data": {
        "left_knee_flex_ext": [angle_frame0, angle_frame1, ...],
        "right_knee_flex_ext": [...],
        ...
      }
    }
    """
    csv_path = find_file(tracker_dir / "joint_angles", "*joint_angles.csv", tracker_id)
    if not csv_path:
        return None

    print(f"  Loading joint angles from {csv_path.name} ...")

    joints = set()
    components = set()
    sides = set()
    # Collect into {key: {frame: angle}}
    raw: dict[str, dict[int, float]] = defaultdict(dict)
    max_frame = 0

    with open(csv_path, "r", newline="") as f:
        for row in csv.DictReader(f):
            frame = int(row["frame"])
            side = row["side"].strip().lower()
            joint = row["joint"].strip().lower()
            component = row["component"].strip().lower()
            angle = float(row["angle"])

            key = f"{side}_{joint}_{component}"
            raw[key][frame] = angle

            joints.add(joint)
            components.add(component)
            sides.add(side)
            if frame > max_frame:
                max_frame = frame

    # Convert to dense arrays (None for missing frames)
    data = {}
    for key, frame_map in raw.items():
        arr = [None] * (max_frame + 1)
        for frame, angle in frame_map.items():
            arr[frame] = angle
        data[key] = arr

    print(f"    Joints: {sorted(joints)}")
    print(f"    Components: {sorted(components)}")
    print(f"    {len(data)} series, {max_frame + 1} frames")

    return {
        "joints": sorted(joints),
        "components": sorted(components),
        "sides": sorted(sides),
        "data": data,
    }


# ════════════════════════════════════════════════════════════════════
# Skeleton Loading
# ════════════════════════════════════════════════════════════════════

def convert_connections(segment_connections: dict, landmark_names: list[str]):
    valid = set(landmark_names)
    return [
        [seg["proximal"], seg["distal"]]
        for seg in segment_connections.values()
        if seg.get("proximal") in valid and seg.get("distal") in valid
    ]


def find_landmark_index(landmarks: list[str], target: str) -> int | None:
    if target in landmarks:
        return landmarks.index(target)
    target_n = target.lower().replace("_", " ")
    for i, name in enumerate(landmarks):
        if target_n in name.lower().replace("_", " "):
            return i
    return None


# ════════════════════════════════════════════════════════════════════
# Main loader — assembles all layers
# ════════════════════════════════════════════════════════════════════

def load_all_data(recording: Path, tid: str) -> tuple[dict, dict]:
    """
    Returns:
      skeleton_payload: skeleton + gait events (sent as /gait_data.json)
      layers: {layer_name: payload_bytes} for additional layer endpoints
    """
    tracker_dir = recording / "validation" / tid
    print(f"Loading {tid} from {tracker_dir} ...")

    # Skeleton
    h = Human.from_data(tracker_dir)
    landmarks = h.body.anatomical_structure.landmark_names
    connections = convert_connections(h.body.anatomical_structure.segment_connections, landmarks)
    positions = h.body.xyz.as_array
    print(f"  Shape: {positions.shape}  ({len(landmarks)} landmarks)")

    landmark_indices = {
        "left_heel": find_landmark_index(landmarks, LEFT_HEEL),
        "right_heel": find_landmark_index(landmarks, RIGHT_HEEL),
        "left_toe": find_landmark_index(landmarks, LEFT_TOE),
        "right_toe": find_landmark_index(landmarks, RIGHT_TOE),
    }

    # Discover available layers
    available_layers = []
    layers = {}

    # Gait events
    gait_events = load_gait_events(tracker_dir)
    if gait_events:
        available_layers.append({
            "id": "gait_events",
            "label": "Gait Events",
            "dataUrl": "/layers/gait_events.json",
        })
        layers["gait_events"] = json.dumps(gait_events).encode("utf-8")

    # Joint angles
    joint_angles = load_joint_angles(tracker_dir)
    if joint_angles:
        available_layers.append({
            "id": "joint_angles",
            "label": "Joint Angles",
            "dataUrl": "/layers/joint_angles.json",
        })
        layers["joint_angles"] = json.dumps(joint_angles).encode("utf-8")

    # Build main payload (skeleton + metadata)
    skeleton_payload = {
        "tracker": getattr(h, "tracker", None) or tid,
        "positions": positions.tolist(),
        "connections": connections,
        "landmarks": landmarks,
        "landmark_indices": landmark_indices,
        "fps": FRAMERATE,
        "color": SKELETON_COLOR,
        "available_layers": available_layers,
        # Keep gait_events inline for backward compat with existing viewer
        "gait_events": gait_events or {"heel_strikes": [], "toe_offs": []},
    }

    return skeleton_payload, layers


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

    skeleton_payload, layers = load_all_data(recording, tracker_id)
    main_bytes = json.dumps(skeleton_payload).encode("utf-8")

    n_hs = len(skeleton_payload["gait_events"]["heel_strikes"])
    n_to = len(skeleton_payload["gait_events"]["toe_offs"])
    n_frames = len(skeleton_payload["positions"])
    n_layers = len(skeleton_payload["available_layers"])

    print(f"\n  Main payload: {len(main_bytes) / 1024 / 1024:.1f} MB")
    for name, b in layers.items():
        print(f"  Layer '{name}': {len(b) / 1024 / 1024:.1f} MB")

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(viewer_dir), **kwargs)

        def do_GET(self):
            path = self.path.split("?", 1)[0]

            if path == "/gait_data.json":
                self._serve_json(main_bytes)
                return

            if path.startswith("/layers/") and path.endswith(".json"):
                layer_id = path.removeprefix("/layers/").removesuffix(".json")
                if layer_id in layers:
                    self._serve_json(layers[layer_id])
                    return
                self.send_error(404, f"Unknown layer: {layer_id}")
                return

            return super().do_GET()

        def _serve_json(self, data: bytes):
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format, *args):
            if any(kw in str(args) for kw in ("gait_data", "layers/", "404")):
                super().log_message(format, *args)

    port = pick_free_port()
    host = "127.0.0.1"
    url = f"http://{host}:{port}/gait_events_viewer.html"

    print(f"\n{'═' * 60}")
    print(f"  Gait Visualization Server")
    print(f"  Recording:  {recording}")
    print(f"  Tracker:    {tracker_id}")
    print(f"  Frames:     {n_frames} @ {FRAMERATE} Hz")
    print(f"  Events:     {n_hs} HS, {n_to} TO")
    print(f"  Layers:     {n_layers} available")
    for layer in skeleton_payload["available_layers"]:
        print(f"    · {layer['label']} ({layer['id']})")
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