import sys, toml
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

AXIS_LEN_MM = 400          # length of camera Z-axis arrow
BOARD_FRAME = 50           # 0-based index ⇒ “20-th frame”


# ───────────────────────────────────────────────────── camera helpers
def load_camera_poses(calib):
    """
    Returns two dicts keyed 'cam_0', …:
        centres_world   – {cam: (3,) np.array}
        R_cam2world     – {cam: (3,3) np.array}
    Works with both the new per-camera fields and the older metadata lists.
    """
    centres, R_wc = {}, {}

    # ── preferred: pull directly from each cam_N table ─────────────
    cam_keys = sorted(k for k in calib if k.startswith("cam_"))
    got_per_cam = all("world_position" in calib[k] for k in cam_keys)

    if got_per_cam:
        for cam in cam_keys:
            centres[cam] = np.array(calib[cam]["world_position"], dtype=float)
            R_wc[cam]    = np.array(calib[cam]["world_orientation"], dtype=float)
    return centres, R_wc


def draw_cameras(ax, centres, R_cam2world):
    for centre in centres.values():
        ax.scatter(*centre, color="k", s=40)

    for centre, R in zip(centres.values(), R_cam2world.values()):
        end = centre + R[:, 2] * AXIS_LEN_MM      # camera +Z
        ax.plot(*zip(centre, end), color="orange", lw=2)


# ───────────────────────────────────────────────────── board helper
def load_board_points(path):
    xyz = np.load(path)               # shape: (n_frames, n_pts, 3)
    if BOARD_FRAME >= xyz.shape[0]:
        raise IndexError(f"BOARD_FRAME={BOARD_FRAME} exceeds array length.")
    return xyz[BOARD_FRAME]           # (n_pts, 3)


def draw_board(ax, board_pts):
    ax.scatter(board_pts[:, 0], board_pts[:, 1], board_pts[:, 2],
               color="crimson", s=12, depthshade=False, label="Charuco pts")


# ───────────────────────────────────────────────────── total plotting
def plot_all(centres, R_cam2world, board_pts):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    draw_cameras(ax, centres, R_cam2world)
    draw_board(ax, board_pts)

    # ───── equal-ish scale: mean ± 500 mm across ALL plotted XYZ
    all_xyz = np.vstack([*centres.values(), board_pts])
    xyz_mean = all_xyz.mean(axis=0)
    for set_lim, m in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), xyz_mean):
        set_lim(m - 500, m + 500)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    plt.title("Cameras, Look Directions, and ChArUco Board (20-th frame)")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ───────────────────────────────────────────────────── CLI wrapper
def main(toml_path, board_path):
    with open(toml_path, "r") as f:
        calib = toml.load(f)

    centres, R_wc = load_camera_poses(calib)
    board_pts = load_board_points(board_path)
    plot_all(centres, R_wc, board_pts)



if __name__ == "__main__":
    path_to_toml = Path(r"D:\2025_07_31_JSM_pilot\freemocap\2025-07-31_14-50-06_GMT-4_jsm_pilot_calibration\2025-07-31_14-50-06_GMT-4_jsm_pilot_calibration_camera_calibration.toml")
    path_to_charuco = Path(r"D:\2025_07_31_JSM_pilot\freemocap\2025-07-31_16-00-42_GMT-4_jsm_nih_trial_1\output_data\mediapipe_body_3d_xyz.npy")
    # path_to_charuco = Path(r"D:\2025_07_31_JSM_pilot\freemocap\2025-07-31_14-57-18_GMT-4_jsm_pilot_calibration\output_data\charuco_3d_xyz.npy")
    main(path_to_toml, path_to_charuco)
