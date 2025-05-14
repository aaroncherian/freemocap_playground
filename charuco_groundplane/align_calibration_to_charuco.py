import numpy as np
import cv2
import toml
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt



def create_3d_space_plot(camera_3d_locations: List[np.ndarray], charuco_3d_frame: np.ndarray,
                         origin: np.ndarray, basis: tuple[np.ndarray, np.ndarray, np.ndarray],
                         title: str = "") -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    camera_markers = ['o', 's', '^']

    for cam in camera_3d_locations:
        ax.scatter(*cam, color='blue', marker='o')

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

def get_unit_vector(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)

def compute_basis(charuco_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_vec = charuco_frame[4] - charuco_frame[0]
    y_vec = charuco_frame[3] - charuco_frame[0]

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
    charuco_frame = charuco_3d[1440, :, :]

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
    cs =  [get_camera_loc_in_world_space(info['rmatrix'], info['tvec'])
        for key, info in sorted(camera_info_dict.items())]
    
    create_3d_space_plot(cs, charuco_frame, np.zeros(3), (np.eye(3)[:, 0], np.eye(3)[:, 1], np.eye(3)[:, 2]), title="Original Frame")

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

    new_cs = [
        get_camera_loc_in_world_space(info['new_rmat'], info['new_tvec'])
        for key, info in sorted(camera_info_dict.items())]
    

    charuco_frame_aligned = (rotation_matrix.T @ (charuco_frame - origin).T).T
    create_3d_space_plot(new_cs, charuco_frame_aligned, np.zeros(3), (np.eye(3)[:, 0], np.eye(3)[:, 1], np.eye(3)[:, 2]), title="Aligned Frame")


    # Update and save new calibration
    new_calibration = calibration.copy()
    for cam_key, info in camera_info_dict.items():
        rvec_new, _ = cv2.Rodrigues(info['new_rmat'])
        new_calibration[cam_key]['rotation'] = rvec_new.flatten().tolist()
        new_calibration[cam_key]['translation'] = info['new_tvec'].flatten().tolist()




    # with open(path_to_output_toml, 'w') as f:
    #     toml.dump(new_calibration, f)

if __name__ == "__main__":
    base_dir = Path(r"D:\2025-04-28-calibration")
    align_calibration_to_charuco(
        path_to_input_toml=base_dir / "2025-04-28-calibration_camera_calibration_aligned.toml",
        path_to_charuco_npy=base_dir / "output_data" / "aligned_charuco_3d.npy",
        path_to_output_toml=base_dir / "2025-04-28-calibration_camera_calibration_aligned.toml"
    )
