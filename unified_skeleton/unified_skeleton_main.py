
from pathlib import Path
import numpy as np

# from model_info import (
#     mediapipe_body,
#     virtual_markers,
#     segment_connections,
#     center_of_mass_anthropometric_data,
#     joint_hierarchy,
# )

from openpose_model_info import (
    landmark_names,
    virtual_markers_definitions,
    segment_connections,
    center_of_mass_definitions,
    joint_hierarchy
)

from model_utils import create_skeleton_model
from skeleton_utils import integrate_freemocap_data_into_skeleton_model

from center_of_mass.skeleton_center_of_mass import calculate_center_of_mass_from_skeleton
from enforce_rigid_bones.skeleton_rigid_bones import enforce_rigid_bones_from_skeleton


path_to_session_folder = Path(r"D:\steen_pantsOn_gait_3_cameras")
path_to_data_folder = path_to_session_folder / 'output_data'/'openpose_data'
path_to_data = path_to_data_folder / 'openpose_body_hands_face_xyz.npy'


skeleton = create_skeleton_model(
    actual_markers=landmark_names,
    virtual_markers=virtual_markers_definitions,
    segment_connections=segment_connections,
    joint_hierarchy=joint_hierarchy,
    anthropometric_data = center_of_mass_definitions
)

## Some examples of how to access different things from the skeleton
print(f'Marker Names: {skeleton.markers.original_marker_names}') #actual marker names
print(f'Virtual Markers: {skeleton.markers.virtual_markers.virtual_markers}') #virtual markers + weights
print(f'All Markers: {skeleton.markers.all_markers}') #list of all markers (actual + virtual, if virtual is included)

print(f'Segment Connections: {skeleton.segments}') #segment connections
print(f"Proximal marker for left upper arm: {skeleton.segments['left_upper_arm'].proximal}") #proximal and distal markers for a segment

print(f'Joint Hierarchy: {skeleton.joint_hierarchy}') #joint hierarchy
print(f'Anthropometric Data: {skeleton.anthropometric_data}') #anthropometric data


freemocap_data = np.load(path_to_data)
skeleton_3d = integrate_freemocap_data_into_skeleton_model(skeleton, freemocap_data)

#Accessing 3d data from the skeleton
#for a marker
print(f"3D Data for left shoulder: {skeleton_3d.marker_data['left_shoulder']}") #3d data for a marker (actual or virtual)
#for a segment
print(f"3D Data for left upper arm: {skeleton_3d.get_segment_markers('left_upper_arm')}") #3d data for proximal and distal markers of a segment

total_body_center_of_mass = calculate_center_of_mass_from_skeleton(skeleton=skeleton_3d)

rigid_bone_data = enforce_rigid_bones_from_skeleton(skeleton=skeleton_3d)

f = 2