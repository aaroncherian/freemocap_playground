import numpy as np
import cv2
import toml
from pathlib import Path
from typing import Dict

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

def get_camera_loc_in_world_space(rmatrix: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    return -rmatrix.T @ tvec

def align_calibration_to_charuco(
    path_to_input_toml: Path,
    path_to_charuco_npy: Path,
    path_to_output_toml: Path
) -> None:
    # Load calibration and charuco data
    with open(path_to_input_toml, 'r') as f:
        calibration = toml.load(f)

    charuco_3d = np.load(path_to_charuco_npy)
    charuco_frame = charuco_3d[-1, :, :]

    # Parse camera extrinsics
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

    # Compute basis and transform
    x_hat, y_hat, z_hat = compute_basis(charuco_frame)
    origin = charuco_frame[0]
    rotation_matrix = np.column_stack([x_hat, y_hat, z_hat])

    for key, info in camera_info_dict.items():
        tvec = info['tvec']
        rmat = info['rmatrix']

        t_delta = rmat @ origin
        new_tvec = tvec + t_delta
        new_rmat = rmat @ rotation_matrix

        camera_info_dict[key]['new_tvec'] = new_tvec
        camera_info_dict[key]['new_rmat'] = new_rmat

    # Update and save new calibration
    new_calibration = calibration.copy()
    for cam_key, info in camera_info_dict.items():
        rvec_new, _ = cv2.Rodrigues(info['new_rmat'])
        new_calibration[cam_key]['rotation'] = rvec_new.flatten().tolist()
        new_calibration[cam_key]['translation'] = info['new_tvec'].flatten().tolist()

    with open(path_to_output_toml, 'w') as f:
        toml.dump(new_calibration, f)

# Example usage
if __name__ == "__main__":
    base_dir = Path(__file__).parent
    align_calibration_to_charuco(
        path_to_input_toml=base_dir / 'data' / 'original_walk' / "2025-04-23_19-01-55-517Z_atc_test_calibration_camera_calibration.toml",
        path_to_charuco_npy=base_dir / 'data' / 'original_walk' / "charuco_3d.npy",
        path_to_output_toml=base_dir / 'data' / 'aligned_walk' / "aligned_camera_calibration_v2.toml"
    )
