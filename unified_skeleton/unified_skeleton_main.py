from model_info import (
    mediapipe_body,
    virtual_markers,
    segment_connections,
    center_of_mass_anthropometric_data,
    joint_hierarchy,
)

from model_utils import create_skeleton_model
from skeleton_utils import integrate_freemocap_data_into_skeleton_model
from models.skeleton import Skeleton

from center_of_mass.skeleton_center_of_mass import calculate_center_of_mass_from_skeleton

from enforce_rigid_bones.skeleton_rigid_bones import enforce_rigid_bones_from_skeleton

from pathlib import Path
import numpy as np




path_to_session_folder = Path(r'D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_13_48_44_MDN_treadmill_2')
path_to_data_folder = path_to_session_folder / 'output_data'
path_to_data = path_to_data_folder / 'mediapipe_body_3d_xyz.npy'



skeleton = create_skeleton_model(
    actual_markers=mediapipe_body,
    virtual_markers=virtual_markers,
    segment_connections=segment_connections,
    joint_hierarchy=joint_hierarchy,
    anthropometric_data = center_of_mass_anthropometric_data
)


freemocap_data = np.load(path_to_data)
skeleton_3d = integrate_freemocap_data_into_skeleton_model(skeleton, freemocap_data)

total_body_center_of_mass = calculate_center_of_mass_from_skeleton(skeleton=skeleton_3d)
rigid_bone_data = enforce_rigid_bones_from_skeleton(skeleton=skeleton_3d)

f = 2