from skellymodels.managers.human import Human
from skellymodels.models.tracking_model_info import MediapipeModelInfo

from scipy.signal import find_peaks

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
def calculate_spherical_angles(human: Human):
    body_3d_xyz = human.body.trajectories['3d_xyz']
    spine_vector = body_3d_xyz.segment_data(human.body.anatomical_structure.segment_connections)['spine']['proximal'] - body_3d_xyz.segment_data(human.body.anatomical_structure.segment_connections)['spine']['distal']
    spine_vector_magnitude = np.linalg.norm(spine_vector, axis=1)
    spine_vector_azimuthal = np.arctan2(spine_vector[:, 1], spine_vector[:, 0])
    spine_vector_polar = np.arccos(spine_vector[:,2]/(spine_vector_magnitude + 1e-9))

    return spine_vector_azimuthal, spine_vector_polar, spine_vector_magnitude


path_to_recording = Path(r'C:\Users\Matthis Lab\skellycam_data\recordings\2025-12-06_12-58-43_GMT-5_OK_stoop')
path_to_output_data = path_to_recording / 'output_data'/     'mediapipe_skeleton_3d.npy'

# model_info = ModelInfo(config_path = Path(__file__).parent/'mediapipe_just_body.yaml')
human = Human.from_tracked_points_numpy_array(
    name = 'human', 
    model_info=MediapipeModelInfo(),
    tracked_points_numpy_array=np.load(path_to_output_data)
)
num_frames = human.body.xyz.num_frames

spine_vector_azimuthal, spine_vector_polar, spine_vector_magnitude = calculate_spherical_angles(
    human=human
)

spine_vector_polar = np.degrees(spine_vector_polar)

peaks,_ = find_peaks(spine_vector_polar, height = 20)

x = list(range(num_frames))



def calculate_trunk_inclination(path_to_recording:Path):
    path_to_output_data = path_to_recording / 'output_data'/     'mediapipe_skeleton_3d.npy'

    human = Human.from_tracked_points_numpy_array(
        name = 'human', 
        model_info=MediapipeModelInfo(),
        tracked_points_numpy_array=np.load(path_to_output_data)
    )
    num_frames = human.body.xyz.num_frames

    spine_vector_azimuthal, spine_vector_polar, spine_vector_magnitude = calculate_spherical_angles(
        human=human
    )

    spine_vector_polar = np.degrees(spine_vector_polar)

    return spine_vector_polar


stoop_angles = calculate_trunk_inclination(path_to_recording = Path(r'C:\Users\Matthis Lab\skellycam_data\recordings\2025-12-06_12-58-43_GMT-5_OK_stoop'))

squat_angles = calculate_trunk_inclination(path_to_recording=Path(r"C:\Users\Matthis Lab\skellycam_data\recordings\2025-12-06_12-54-48_GMT-5_OK_squat"))


stoop_peaks,_ = find_peaks(stoop_angles, height = 20, distance=20)
squat_peaks, _ = find_peaks(squat_angles, height=20, distance = 20)

fig = plt.figure()
fig, (ax1, ax2) = plt.subplots(1,2)

ax1.plot(stoop_angles)
ax2.plot(squat_angles)

ylim = 100
ax1.set_ylim([0,ylim])
ax2.set_ylim([0,ylim])

ax1.scatter(stoop_peaks, stoop_angles[stoop_peaks], color = 'red')
ax2.scatter(squat_peaks, squat_angles[squat_peaks], color = 'red')

ax1.set_title("Stoop Form")
ax2.set_title("Squat Form")

ax1.set_xlabel("Time (frames)")
ax2.set_xlabel("Time (frames)")

ax1.set_ylabel("Trunk inclination (degrees)")

plt.show()
f = 2