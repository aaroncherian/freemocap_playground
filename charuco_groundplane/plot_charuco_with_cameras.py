import numpy as np
import matplotlib.pyplot as plt
import cv2
import toml

from pathlib import Path
from typing import List, Dict

def get_unit_vector(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)

def compute_basis(charuco_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_vec = charuco_frame[18] - charuco_frame[0]
    y_vec = charuco_frame[5] - charuco_frame[0]

    x_hat = get_unit_vector(x_vec)
    y_hat_raw = get_unit_vector(y_vec)
    z_hat = get_unit_vector(np.cross(x_hat, y_hat_raw))
    y_hat = get_unit_vector(np.cross(z_hat, x_hat))

    return x_hat, y_hat, z_hat

def create_3d_space_plot(camera_3d_locations: List[np.ndarray], charuco_3d_frame: np.ndarray,
                         origin: np.ndarray, basis: tuple[np.ndarray, np.ndarray, np.ndarray],
                         title: str = "") -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    camera_markers = ['o', 's', '^']

    for cam, marker in zip(camera_3d_locations, camera_markers):
        ax.scatter(*cam, color='blue', marker=marker)

    ax.scatter(charuco_3d_frame[:, 0], charuco_3d_frame[:, 1], charuco_3d_frame[:, 2])

    x_hat, y_hat, z_hat = basis
    length = 500
    ax.quiver(*origin, *(x_hat * length), color='red', label='X')
    ax.quiver(*origin, *(y_hat * length), color='green', label='Y')
    ax.quiver(*origin, *(z_hat * length), color='blue', label='Z')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title(title)
    plt.show()

def get_camera_loc_in_world_space(rmatrix: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    return -rmatrix.T @ tvec

# Paths
base_dir = Path(__file__).parent
path_to_toml = base_dir / 'data' / 'original_walk' / "2025-04-23_19-01-55-517Z_atc_test_calibration_camera_calibration.toml"
path_to_3d_charuco_data = base_dir / 'data' / 'original_walk' / "charuco_3d.npy"
path_to_aligned_toml = base_dir / 'data' / 'aligned_walk' / "aligned_camera_calibration_v2.toml"

# Load data
with open(path_to_toml, 'r') as f:
    calibration = toml.load(f)

charuco_3d = np.load(path_to_3d_charuco_data)
charuco_frame = charuco_3d[-1, :, :]

# Parse calibration
camera_info_dict: Dict[str, Dict[str, np.ndarray]] = {}
for key, val in calibration.items():
    if key == 'metadata':
        continue
    rmat, _ = cv2.Rodrigues(np.array(val['rotation']))
    camera_info_dict[key] = {
        'rvec': np.array(val['rotation']),
        'rmatrix': rmat,
        'tvec': np.array(val['translation'])
    }

original_cs = [
    get_camera_loc_in_world_space(info['rmatrix'], info['tvec'])
    for key, info in sorted(camera_info_dict.items())
]

# Compute and plot original coordinate frame
x_hat, y_hat, z_hat = compute_basis(charuco_frame)
origin = charuco_frame[0]
create_3d_space_plot(original_cs, charuco_frame, origin, (x_hat, y_hat, z_hat), title="Original Frame")

# Build rotation matrix
rotation_matrix = np.column_stack([x_hat, y_hat, z_hat])

# Update camera extrinsics
for key, info in camera_info_dict.items():
    tvec = info['tvec']
    rmat = info['rmatrix']

    t_delta = rmat @ origin
    new_tvec = tvec + t_delta
    new_rmat = rmat @ rotation_matrix

    camera_info_dict[key]['new_tvec'] = new_tvec
    camera_info_dict[key]['new_rmat'] = new_rmat

# Compute new camera locations
new_cs = [
    get_camera_loc_in_world_space(info['new_rmat'], info['new_tvec'])
    for key, info in sorted(camera_info_dict.items())
]

# Transform Charuco frame
charuco_aligned = (rotation_matrix.T @ (charuco_frame - origin).T).T
create_3d_space_plot(new_cs, charuco_aligned, np.zeros(3), (np.eye(3)[:, 0], np.eye(3)[:, 1], np.eye(3)[:, 2]), title="Aligned Frame")

# Save new calibration
new_calibration = calibration.copy()
for cam_key, info in camera_info_dict.items():
    rvec_new, _ = cv2.Rodrigues(info['new_rmat'])
    new_calibration[cam_key]['rotation'] = rvec_new.flatten().tolist()
    new_calibration[cam_key]['translation'] = info['new_tvec'].flatten().tolist()

with open(path_to_aligned_toml, 'w') as f:
    toml.dump(new_calibration, f)

# Diagnostics
d_old = np.linalg.norm(original_cs[0] - original_cs[1])
d_new = np.linalg.norm(new_cs[0] - new_cs[1])
print(f"Old inter-camera distance: {d_old:.2f}")
print(f"New inter-camera distance: {d_new:.2f}")
