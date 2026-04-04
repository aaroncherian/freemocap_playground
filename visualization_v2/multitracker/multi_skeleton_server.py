"""
Multi-Skeleton Comparison Server
=================================
Loads skeleton data from multiple trackers and serves them
side-by-side to the HTML viewer.

Each tracker's skeleton is offset along the X axis so they
appear next to each other rather than overlaid.

The viewer also shows position trajectories for selected
landmarks, overlaid from all trackers.

Usage:
    python multi_skeleton_server.py
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
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════

path_to_recording = Path(
    r"D:\validation\data\2026-01-30-JTM\2026-01-30_11-21-06_GMT-5_JTM_treadmill_1"
)

candidate_trackers = ["mediapipe", "vitpose", "qualisys", "rtmpose"]

path_to_viewer_folder = Path(
    r"C:\Users\aaron\Documents\GitHub\freemocap_playground\visualization_v2\multitracker"
)

FRAMERATE = 30
X_SPACING = 1500  # mm between each skeleton along X axis


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


def load_tracker(recording: Path, tracker_id: str) -> dict | None:
    """Load a single tracker's skeleton data. Returns None if not found."""
    tracker_dir = recording / "validation" / tracker_id
    if not tracker_dir.exists():
        print(f"  ⚠ {tracker_id}: directory not found, skipping")
        return None

    try:
        h = Human.from_data(tracker_dir)
    except Exception as e:
        print(f"  ⚠ {tracker_id}: failed to load — {e}")
        return None

    landmarks = h.body.anatomical_structure.landmark_names
    connections = convert_connections(
        h.body.anatomical_structure.segment_connections, landmarks
    )
    positions = h.body.xyz.as_array  # (frames, landmarks, 3)

    print(f"  ✓ {tracker_id}: {positions.shape[0]} frames, {len(landmarks)} landmarks")

    return {
        "tracker": tracker_id,
        "positions": positions,  # keep as numpy for now
        "connections": connections,
        "landmarks": landmarks,
    }


# ════════════════════════════════════════════════════════════════════
# Build the payload
# ════════════════════════════════════════════════════════════════════

# Distinct colors for each tracker
TRACKER_COLORS = {
    "qualisys":  {"hex": 0x000000, "css": "#000000"},  # black (reference)
    "mediapipe": {"hex": 0x2f6efc, "css": "#2f6efc"},  # blue
    "rtmpose":   {"hex": 0xff7f0e, "css": "#ff7f0e"},  # orange
    "vitpose":   {"hex": 0x2ca02c, "css": "#2ca02c"},  # green
}

# Fallback colors if tracker name isn't in the map
FALLBACK_COLORS = [
    {"hex": 0xD64045, "css": "#D64045"},
    {"hex": 0x1B998B, "css": "#1B998B"},
    {"hex": 0xC17817, "css": "#C17817"},
    {"hex": 0x5F4690, "css": "#5F4690"},
]


def build_payload(recording: Path, trackers: list[str]) -> bytes:
    """Load all trackers, compute offsets, build JSON payload."""
    print(f"Loading trackers from {recording} ...")

    loaded = []
    for tid in trackers:
        result = load_tracker(recording, tid)
        if result is not None:
            loaded.append(result)

    if not loaded:
        raise RuntimeError("No trackers loaded successfully!")

    # Find the common frame count (use minimum across all trackers)
    min_frames = min(d["positions"].shape[0] for d in loaded)
    print(f"\n  Common frames: {min_frames}")

    # Compute centroid of first tracker at frame 0 to use as reference
    ref_positions = loaded[0]["positions"][0]
    ref_centroid_x = float(np.nanmean(ref_positions[:, 0]))

    # Build skeleton entries with X offsets
    skeletons = []
    for i, d in enumerate(loaded):
        tid = d["tracker"]
        color = TRACKER_COLORS.get(tid, FALLBACK_COLORS[i % len(FALLBACK_COLORS)])

        # Compute this tracker's centroid at frame 0
        frame0 = d["positions"][0]
        centroid_x = float(np.nanmean(frame0[:, 0]))

        # Offset: space them out along X, centered around the reference
        x_offset = i * X_SPACING - centroid_x + ref_centroid_x

        # Apply offset to positions
        positions = d["positions"][:min_frames].copy()
        positions[:, :, 0] += x_offset

        skeletons.append({
            "tracker": tid,
            "positions": positions.tolist(),
            "connections": d["connections"],
            "landmarks": d["landmarks"],
            "color": color,
            "x_offset": x_offset,
        })

    # Build a shared landmark map for trajectory plots
    # Find landmarks that exist in ALL trackers
    all_landmark_sets = [set(d["landmarks"]) for d in loaded]
    common_landmarks = sorted(set.intersection(*all_landmark_sets))

    payload = {
        "skeletons": skeletons,
        "fps": FRAMERATE,
        "total_frames": min_frames,
        "common_landmarks": common_landmarks,
        "x_spacing": X_SPACING,
    }

    return json.dumps(payload).encode("utf-8")


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

    viewer_html = "multi_skeleton_viewer.html"
    if not (viewer_dir / viewer_html).exists():
        raise FileNotFoundError(
            f"Missing {viewer_html} in {viewer_dir}\n"
            f"Copy the HTML file there first!"
        )

    payload_bytes = build_payload(recording, candidate_trackers)

    print(f"\n  Payload size: {len(payload_bytes) / 1024 / 1024:.1f} MB")

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(viewer_dir), **kwargs)

        def do_GET(self):
            path = self.path.split("?", 1)[0]

            if path == "/skeleton_data.json":
                self._serve_json(payload_bytes)
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
            if any(kw in str(args) for kw in ("skeleton_data", "404")):
                super().log_message(format, *args)

    port = pick_free_port()
    host = "127.0.0.1"
    url = f"http://{host}:{port}/{viewer_html}"

    # Count loaded trackers
    import json as _json
    info = _json.loads(payload_bytes)
    n_skeletons = len(info["skeletons"])
    tracker_names = [s["tracker"] for s in info["skeletons"]]

    print(f"\n{'═' * 60}")
    print(f"  Multi-Skeleton Comparison Server")
    print(f"  Recording:  {recording}")
    print(f"  Trackers:   {', '.join(tracker_names)} ({n_skeletons})")
    print(f"  Frames:     {info['total_frames']} @ {FRAMERATE} Hz")
    print(f"  Spacing:    {X_SPACING} mm along X")
    print(f"  Common landmarks: {len(info['common_landmarks'])}")
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