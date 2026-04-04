"""
Balance Visualization Server
==============================
Loads skeleton data + COM balance data from the path_length_analysis folder.

Expected files in {tracker_dir}/path_length_analysis/{analysis_folder}/:
  - condition_data.json        → frame intervals + total path lengths
  - condition_positions.csv    → COM XYZ per condition per frame
  - condition_velocities.csv   → COM velocity XYZ per condition per frame

Usage:
    python balance_server.py
"""

from __future__ import annotations

import json
import socket
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

from skellymodels.managers.human import Human


# ════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════

path_to_recording = Path(
    r"D:\validation\data\2025_07_31_JSM_pilot\freemocap\2025-07-31_16-00-42_GMT-4_jsm_nih_trial_1"
)

tracker_id = "rtmpose"
analysis_folder = "analysis_2026-03-11_15_04_36"

path_to_viewer_folder = Path(
    r"C:\Users\aaron\Documents\GitHub\freemocap_playground\visualization_v2\balance"
)

FRAMERATE = 30

CONDITIONS = [
    "Eyes Open/Solid Ground",
    "Eyes Closed/Solid Ground",
    "Eyes Open/Foam",
    "Eyes Closed/Foam",
]

SHORT_LABELS = ["EO / Solid", "EC / Solid", "EO / Foam", "EC / Foam"]


# ════════════════════════════════════════════════════════════════════
# COM + Ellipse
# ════════════════════════════════════════════════════════════════════

def compute_prediction_ellipse_95(x, y):
    """Compute 95% prediction ellipse for 2D data."""
    cov_matrix = np.cov(x, y)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    chi2_val = chi2.ppf(0.95, df=2)

    a = np.sqrt(eigenvalues[0] * chi2_val)
    b = np.sqrt(eigenvalues[1] * chi2_val)
    area = np.pi * a * b

    theta = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])

    t = np.linspace(0, 2 * np.pi, 100)
    ellipse_x = a * np.cos(t)
    ellipse_y = b * np.sin(t)

    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    ellipse_points = R @ np.array([ellipse_x, ellipse_y])

    return {
        "ellipse_x": ellipse_points[0].tolist(),
        "ellipse_y": ellipse_points[1].tolist(),
        "area": float(area),
        "semi_major": float(a),
        "semi_minor": float(b),
    }


def compute_cumulative_path_length(x, y):
    """Compute cumulative path length at each frame."""
    dx = np.diff(x)
    dy = np.diff(y)
    step_lengths = np.sqrt(dx**2 + dy**2)
    return np.concatenate([[0], np.cumsum(step_lengths)]).tolist()


def load_balance_data(analysis_dir: Path) -> tuple[dict | None, list | None]:
    """
    Load all three balance analysis files and build per-condition data.

    Returns:
        conditions_data: dict keyed by condition name
        frame_ranges: list of [start, end] per condition
    """
    # Load condition_data.json for frame intervals and path lengths
    cond_json_path = analysis_dir / "condition_data.json"
    if not cond_json_path.exists():
        print(f"  Warning: {cond_json_path} not found")
        return None, None

    with open(cond_json_path) as f:
        cond_json = json.load(f)

    frame_intervals = cond_json.get("Frame Intervals", {})
    total_path_lengths = cond_json.get("Path Lengths:", {})

    # Load positions CSV
    pos_path = analysis_dir / "condition_positions.csv"
    if not pos_path.exists():
        print(f"  Warning: {pos_path} not found")
        return None, None

    pos_df = pd.read_csv(pos_path)
    print(f"  Positions: {pos_df.shape}")

    # Load velocities CSV (optional — we'll include if present)
    vel_path = analysis_dir / "condition_velocities.csv"
    vel_df = None
    if vel_path.exists():
        vel_df = pd.read_csv(vel_path)
        print(f"  Velocities: {vel_df.shape}")

    # Build per-condition data
    conditions_data = {}
    frame_ranges = []

    for i, condition in enumerate(CONDITIONS):
        x_col = f"{condition}_x"
        y_col = f"{condition}_y"
        z_col = f"{condition}_z"

        if x_col not in pos_df.columns:
            print(f"    Warning: '{x_col}' not found, skipping {condition}")
            continue

        x_raw = pos_df[x_col].to_numpy(dtype=float)
        y_raw = pos_df[y_col].to_numpy(dtype=float)
        z_raw = pos_df[z_col].to_numpy(dtype=float) if z_col in pos_df.columns else None

        # Remove NaN
        mask = ~(np.isnan(x_raw) | np.isnan(y_raw))
        x_raw = x_raw[mask]
        y_raw = y_raw[mask]
        if z_raw is not None:
            z_raw = z_raw[mask]

        # Mean-center XY for statokinesiogram
        x_centered = x_raw - np.mean(x_raw)
        y_centered = y_raw - np.mean(y_raw)

        # Ellipse + path length on centered data
        ellipse = compute_prediction_ellipse_95(x_centered, y_centered)
        cumulative_pl = compute_cumulative_path_length(x_centered, y_centered)
        total_pl = total_path_lengths.get(condition, cumulative_pl[-1] if cumulative_pl else 0)

        # Velocities (may have different row count, handle independently)
        vel_data = None
        if vel_df is not None and x_col in vel_df.columns:
            vx = vel_df[x_col].to_numpy(dtype=float)
            vy = vel_df[y_col].to_numpy(dtype=float)
            vel_mask = ~(np.isnan(vx) | np.isnan(vy))
            vx = vx[vel_mask]
            vy = vy[vel_mask]
            vel_mag = np.sqrt(vx**2 + vy**2)
            vel_data = vel_mag.tolist()

        # Frame range from condition_data.json
        interval = frame_intervals.get(condition, [0, len(x_raw) - 1])
        frame_ranges.append(interval)

        conditions_data[condition] = {
            "label": SHORT_LABELS[i],
            "com_x": x_centered.tolist(),
            "com_y": y_centered.tolist(),
            "com_x_raw": x_raw.tolist(),  # uncentered for 3D positioning
            "com_y_raw": y_raw.tolist(),
            "com_z": z_raw.tolist() if z_raw is not None else None,
            "cumulative_path_length": cumulative_pl,
            "total_path_length": float(total_pl),
            "ellipse": ellipse,
            "velocity": vel_data,
            "num_frames": len(x_raw),
        }

        print(f"    {SHORT_LABELS[i]}: {len(x_raw)} frames, "
              f"PL={total_pl:.1f}mm, ellipse={ellipse['area']:.1f}mm², "
              f"skeleton frames {interval[0]}–{interval[1]}")

    return conditions_data, frame_ranges


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


def load_skeleton(tracker_dir: Path) -> dict:
    h = Human.from_data(tracker_dir)
    landmarks = h.body.anatomical_structure.landmark_names
    connections = convert_connections(h.body.anatomical_structure.segment_connections, landmarks)
    positions = h.body.xyz.as_array

    print(f"  Skeleton: {positions.shape} ({len(landmarks)} landmarks)")

    return {
        "positions": positions.tolist(),
        "connections": connections,
        "landmarks": landmarks,
        "total_frames": positions.shape[0],
    }


# ════════════════════════════════════════════════════════════════════
# Main Loader
# ════════════════════════════════════════════════════════════════════

def load_all_data(recording: Path, tid: str) -> bytes:
    tracker_dir = recording / "validation" / tid
    analysis_dir = tracker_dir / "path_length_analysis" / analysis_folder

    print(f"Loading {tid} from {tracker_dir} ...")
    print(f"  Analysis dir: {analysis_dir}")

    skeleton = load_skeleton(tracker_dir)
    balance_data, frame_ranges = load_balance_data(analysis_dir)

    payload = {
        "tracker": tid,
        "fps": FRAMERATE,
        "skeleton": skeleton,
        "conditions": balance_data or {},
        "condition_order": CONDITIONS,
        "short_labels": SHORT_LABELS,
        "frame_ranges": frame_ranges,
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

    if not (viewer_dir / "balance_viewer.html").exists():
        raise FileNotFoundError(
            f"Missing balance_viewer.html in {viewer_dir}\n"
            f"Copy the HTML file there first!"
        )

    payload_bytes = load_all_data(recording, tracker_id)

    print(f"\n  Payload size: {len(payload_bytes) / 1024 / 1024:.1f} MB")

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(viewer_dir), **kwargs)

        def do_GET(self):
            path = self.path.split("?", 1)[0]

            if path == "/balance_data.json":
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload_bytes)))
                self.end_headers()
                self.wfile.write(payload_bytes)
                return

            return super().do_GET()

        def log_message(self, format, *args):
            if any(kw in str(args) for kw in ("balance_data", "404")):
                super().log_message(format, *args)

    port = pick_free_port()
    host = "127.0.0.1"
    url = f"http://{host}:{port}/balance_viewer.html"

    print(f"\n{'═' * 60}")
    print(f"  Balance Visualization Server")
    print(f"  Recording:  {recording}")
    print(f"  Tracker:    {tracker_id}")
    print(f"  Analysis:   {analysis_folder}")
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