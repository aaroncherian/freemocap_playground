from pathlib import Path
import numpy as np

from skellymodels.create_model_skeleton import create_mediapipe_skeleton_model, create_rigid_mediapipe_skeleton_model
from skellymodels.skeleton_models.skeleton import Skeleton
from rigid_bones_com.rigid_bones import enforce_rigid_bones_from_skeleton

def calculate_euclidean_length(marker_a_trajectories, marker_b_trajectories):
    return np.linalg.norm(marker_a_trajectories - marker_b_trajectories, axis=1)

def test_rigid_bones(rigid_bones_data, skeleton_model):
    length_dict = {}
    for segment in skeleton_model.segments:
        proximal_marker = skeleton_model.segments[segment].proximal
        distal_marker = skeleton_model.segments[segment].distal
        lengths = calculate_euclidean_length(skeleton_model.trajectories[distal_marker], skeleton_model.trajectories[proximal_marker])
        length_dict[segment] = lengths

    return length_dict

  

path_to_recording_folder = Path(r'D:\cyr_wheel\cyr_recording')
data_to_use = 'mediapipe_body_3d_xyz.npy'

path_to_output_data = path_to_recording_folder/'output_data'/data_to_use

data = np.load(path_to_output_data)

mediapipe_model = create_mediapipe_skeleton_model()

mediapipe_model.integrate_freemocap_3d_data(data)

rigid_bones_data = enforce_rigid_bones_from_skeleton(mediapipe_model)

rigid_bones_skeleton = create_rigid_mediapipe_skeleton_model()
rigid_bones_skeleton.integrate_freemocap_3d_data(rigid_bones_data)

length_dict = test_rigid_bones(rigid_bones_data, rigid_bones_skeleton)
f = 2
# folder_to_save = path_to_recording_folder/'output_data'/'component_rigid_bones'
# folder_to_save.mkdir(exist_ok=True, parents=True)

# np.save(folder_to_save/'mediapipe_body_3d_xyz.npy', rigid_bones_data)


def apply_rigid_bones(data_to_rigidify:np.ndarray, skeleton_model:Skeleton):
    skeleton_model.integrate_freemocap_3d_data(data_to_rigidify)
    rigid_bones_data = enforce_rigid_bones_from_skeleton(skeleton_model)
    return rigid_bones_data


def calculate_euclidean_length(point_a, point_b):
    return np.linalg.norm(point_a - point_b)


f = 2

