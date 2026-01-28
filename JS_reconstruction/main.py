from __future__ import annotations

import json
import socket
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from skellymodels.managers.human import Human


# ------------------------
# EDIT THESE PATHS
# ------------------------

path_to_recording = Path(
    r"D:\2025_07_31_JSM_pilot\freemocap\2025-07-31_16-35-10_GMT-4_jsm_treadmill_trial_1"
)

# Folder that already contains:
#   index.html
#   viewer.js
#   plots.js
path_to_viewer_folder = Path(
    r"C:\Users\aaron\Documents\GitHub\freemocap_playground\JS_reconstruction"
)

candidate_trackers = ["qualisys", "mediapipe", "vitpose_wholebody", "vitpose_25", "rtmpose"]


# Reference-first palette (ints are 0xRRGGBB)
COLOR_BY_ID = {
    "qualisys": 0x000000,          # black reference
    "mediapipe": 0x2f6efc,         # blue
    "rtmpose": 0xff7f0e,           # orange
    "vitpose_wholebody": 0x2ca02c, # green
    "vitpose_25": 0x9467bd,        # purple
}


def pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def make_payload(h: Human) -> dict:
    return {
        "tracker": getattr(h, "tracker", None),  # optional, if Human provides it
        "positions": h.body.xyz.as_array.tolist(),
        "connections": h.body.anatomical_structure.segment_connections,
        "landmarks": h.body.anatomical_structure.landmark_names,
    }


def build_datasets(recording: Path, tracker_ids: list[str]) -> tuple[bytes, dict[str, bytes]]:
    """
    Returns:
      manifest_bytes: JSON bytes containing datasets list
      payloads: {tracker_id: payload_json_bytes}
    """
    payloads: dict[str, bytes] = {}
    datasets: list[dict] = []
    errors: dict[str, str] = {}

    for tid in tracker_ids:
        p = recording / "validation" / tid
        try:
            h = Human.from_data(p)
            payload = make_payload(h)
            payloads[tid] = json.dumps(payload).encode("utf-8")

            # Prefer Human.tracker (if present) for display, else folder name
            display = payload.get("tracker") or tid

            datasets.append({
                "id": tid,
                "label": str(display),
                "dataUrl": f"/data/{tid}.json",
                "color": int(COLOR_BY_ID.get(tid, 0x888888)),
                "visible": True,
            })
        except Exception as e:
            errors[tid] = f"{type(e).__name__}: {e}"

    # Keep Qualisys first if present (nice mental model: reference first)
    datasets.sort(key=lambda d: (d["id"] != "qualisys", d["id"]))

    manifest = {
        "recording": str(recording),
        "datasets": datasets,
        "errors": errors,  # helpful debug; safe to remove
    }
    return json.dumps(manifest, indent=2).encode("utf-8"), payloads


def run_server():
    viewer_dir = path_to_viewer_folder.resolve()

    # viewer expects static index.html + viewer.js + plots.js :contentReference[oaicite:5]{index=5}
    for fname in ("index.html", "viewer.js", "plots.js"):
        if not (viewer_dir / fname).exists():
            raise FileNotFoundError(f"Missing {fname} in {viewer_dir}")

    recording = path_to_recording.resolve()
    manifest_bytes, payloads = build_datasets(recording, candidate_trackers)

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(viewer_dir), **kwargs)

        def do_GET(self):
            path = self.path.split("?", 1)[0]

            if path == "/manifest.json":
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(manifest_bytes)))
                self.end_headers()
                self.wfile.write(manifest_bytes)
                return

            if path.startswith("/data/") and path.endswith(".json"):
                tid = path.removeprefix("/data/").removesuffix(".json")
                if tid in payloads:
                    b = payloads[tid]
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(b)))
                    self.end_headers()
                    self.wfile.write(b)
                    return
                self.send_error(404, f"Unknown tracker id: {tid}")
                return

            return super().do_GET()

    port = pick_free_port()
    host = "127.0.0.1"
    url = f"http://{host}:{port}/index.html"

    print(f"Viewer folder:   {viewer_dir}")
    print(f"Recording:       {recording}")
    print(f"Manifest:        http://{host}:{port}/manifest.json")
    print(f"Open:            {url}")

    server = ThreadingHTTPServer((host, port), Handler)
    threading.Timer(0.25, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    run_server()