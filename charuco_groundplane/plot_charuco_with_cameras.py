

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cv2
import toml

from pathlib import Path
from typing import List

def get_unit_vector(vector):
    return vector/np.linalg.norm(vector)



def create_3d_space_plot(camera_3d_locations: List[np.ndarray], charuco_3d_frame: np.ndarray):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    camera_markers = ['o', 's', '^']
    for camera, marker in zip(camera_3d_locations,camera_markers):
        ax.scatter(*camera, color='blue', marker=marker)

    ax.scatter(charuco_3d_frame[:, 0], charuco_3d_frame[:, 1], charuco_3d_frame[:, 2])

    # --- Construct orthonormal basis ---
    x_vec = charuco_3d_frame[5] - charuco_3d_frame[0]
    y_vec = charuco_3d_frame[18] - charuco_3d_frame[0]

    x_hat = get_unit_vector(x_vec)
    y_hat_raw = get_unit_vector(y_vec)

    z_hat = get_unit_vector(np.cross(x_hat, y_hat_raw))
    y_hat = get_unit_vector(np.cross(z_hat, x_hat))

    origin = charuco_3d_frame[0]
    length = 500  # adjust as needed to match your scene scale

    # --- Plot the basis vectors ---
    ax.quiver(*origin, *(x_hat * length), color='red', label='X')
    ax.quiver(*origin, *(y_hat * length), color='green', label='Y')
    ax.quiver(*origin, *(z_hat * length), color='blue', label='Z')

    ax.legend()
    plt.show()

path_to_toml = Path(__file__).parents[0]/'data'/'original_walk'/"2025-04-23_19-01-55-517Z_atc_test_calibration_camera_calibration.toml"
path_to_3d_charuco_data = Path(__file__).parents[0]/'data'/'original_walk'/"charuco_3d.npy"

path_to_aligned_toml = Path(__file__).parents[0]/'data'/'aligned_walk'/"aligned_camera_calibration_v2.toml"


with open(path_to_toml, 'r') as f:
    calibration = toml.load(f)

charuco_3d = np.load(path_to_3d_charuco_data)

camera_info_dict = {}
for key, value in calibration.items():
    if key != 'metadata':
        rmat, _ = cv2.Rodrigues(np.array(value['rotation']))
        camera_info_dict[key] = {
            'rvec': np.array(value['rotation']),
            'rmatrix': rmat,
            'tvec': np.array(value['translation'])
        }

f = 2
steps = []

camera_world_space_location = []

#Pc = RPw + t [t is translation shift from world origin to camera origin in camera space]
#Pc = 0 = RPw + t
#t = -RPw = -RC (where C is the coordinates of the camera in world space)
# C = -R.T*t
def get_camera_loc_in_world_space(rmatrix,tvec):
    return -rmatrix.T@tvec

original_cs = [
    get_camera_loc_in_world_space(info['rmatrix'], info['tvec'])
    for key, info in sorted(camera_info_dict.items())
]
steps.append(('world_space', original_cs, 'blue', True))


x_row = [0,1,2,3,4,5]

y_row = [0,6,12,18]

charuco_frame = charuco_3d[-1,:,:]

x_vec = charuco_frame[5,:] - charuco_frame[0,:]
y_vec = charuco_frame[18,:] - charuco_frame[0,:]

create_3d_space_plot(camera_3d_locations=original_cs, charuco_3d_frame=charuco_frame)


x_vec = charuco_frame[5] - charuco_frame[0]
y_vec = charuco_frame[18] - charuco_frame[0]

x_hat = get_unit_vector(x_vec)
y_hat_raw = get_unit_vector(y_vec)

z_hat = get_unit_vector(np.cross(x_hat, y_hat_raw))
y_hat = get_unit_vector(np.cross(z_hat, x_hat))

new_origin = charuco_frame[0]

#if camera 0 is origin, need to shift everything by the 'new origin'
#t = -RC
#t = -R[new origin]
rotation_matrix = np.column_stack([x_hat, y_hat, z_hat])

for key, info in sorted(camera_info_dict.items()):
    tvec = info['tvec']
    rmat = info['rmatrix']

    t_delta = rmat@new_origin

    camera_info_dict[key]['new_tvec'] = tvec + t_delta
    camera_info_dict[key]['new_rmat'] = rmat@rotation_matrix



new_cs = [
    get_camera_loc_in_world_space(info['new_rmat'], info['new_tvec'])
    for key, info in sorted(camera_info_dict.items())
]

charuco_new = (rotation_matrix.T @ (charuco_frame - new_origin).T).T 


create_3d_space_plot(camera_3d_locations=new_cs, charuco_3d_frame=charuco_new)

new_calibration = calibration.copy()

for cam_key, info in camera_info_dict.items():
    r_new = info["new_rmat"]
    t_new = info["new_tvec"]

    rvec_new, _ = cv2.Rodrigues(r_new)  # (3,1)
    new_calibration[cam_key]["rotation"]    = rvec_new.flatten().tolist()
    new_calibration[cam_key]["translation"] = t_new.flatten().tolist()


with open(path_to_aligned_toml, "w") as f:
    toml.dump(new_calibration, f)

d_old = np.linalg.norm(original_cs[0] - original_cs[1])
d_new = np.linalg.norm(new_cs[0]      - new_cs[1])


f = 2
