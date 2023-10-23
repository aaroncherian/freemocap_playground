from pathlib import Path
import numpy as np
from scipy.signal import correlate
from scipy.optimize import minimize
from scipy.stats import mode
import matplotlib.pyplot as plt

# Define paths
path_to_freemocap_recording_folder = Path(r"D:\2023-06-07_TF01\1.0_recordings\treadmill_calib\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1")
path_to_qualisys_data = path_to_freemocap_recording_folder / 'qualisys' / 'resampled_qualisys_joint_centers_3d_xyz.npy'
path_to_freemocap_data = path_to_freemocap_recording_folder / 'output_data' / 'mediapipe_body_3d_xyz.npy'

# Load the data
qualisys_data = np.load(path_to_qualisys_data)
freemocap_data = np.load(path_to_freemocap_data)

# Common markers and their indices in both datasets
common_markers = ['right_hip', 'left_hip', 'right_knee', 'left_knee', 'right_ankle', 'left_ankle', 'right_heel', 'left_heel', 'right_foot_index', 'left_foot_index']
qualisys_indices = common_markers
mediapipe_indices = ['nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye', 'right_eye_outer', 'left_ear', 'right_ear', 'mouth_left', 'mouth_right', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky', 'left_index', 'right_index', 'left_thumb', 'right_thumb', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel', 'right_heel', 'left_foot_index', 'right_foot_index']

qualisys_common_idx = [qualisys_indices.index(marker) for marker in common_markers]
mediapipe_common_idx = [mediapipe_indices.index(marker) for marker in common_markers]

# Calculate velocity magnitude
def calc_velocity_magnitude(data):
    velocity = np.diff(data, axis=0)
    magnitude = np.linalg.norm(velocity, axis=2)
    return magnitude

qualisys_velocity_magnitude = calc_velocity_magnitude(qualisys_data[:, qualisys_common_idx, :])
freemocap_velocity_magnitude = calc_velocity_magnitude(freemocap_data[:, mediapipe_common_idx, :])

# Initialize lists for time shifts
cross_corr_time_shifts = []
optimization_time_shifts = []

# Calculate time shifts
for marker_idx in range(len(common_markers)):
    # Use raw positional data
    qualisys_marker_data = qualisys_data[:, qualisys_common_idx[marker_idx], :]
    freemocap_marker_data = freemocap_data[:, mediapipe_common_idx[marker_idx], :]
    
    # Calculate the Euclidean distance at each frame for both datasets
    qualisys_marker_distance = np.linalg.norm(qualisys_marker_data, axis=1)
    freemocap_marker_distance = np.linalg.norm(freemocap_marker_data, axis=1)

    # Cross-Correlation
    cross_corr = correlate(qualisys_marker_distance, freemocap_marker_distance, mode='full')
    time_shift_cross_corr = cross_corr.argmax() - (len(qualisys_marker_distance) - 1)
    cross_corr_time_shifts.append(time_shift_cross_corr)
    
    # Optimization for Best Fit
    def objective_function(time_shift):
        time_shift = int(time_shift)
        if time_shift >= 0:
            shifted_qualisys = qualisys_marker_distance[time_shift:]
            truncated_freemocap = freemocap_marker_distance[:len(shifted_qualisys)]
        else:
            shifted_qualisys = qualisys_marker_distance[:time_shift]
            truncated_freemocap = freemocap_marker_distance[-time_shift:]
        
        min_length = min(len(shifted_qualisys), len(truncated_freemocap))
        shifted_qualisys = shifted_qualisys[:min_length]
        truncated_freemocap = truncated_freemocap[:min_length]
        
        return np.sum((shifted_qualisys - truncated_freemocap) ** 2)
    
    optimization_result = minimize(objective_function, 0, method='Powell')
    time_shift_optimization = int(optimization_result.x)
    optimization_time_shifts.append(time_shift_optimization)

print("Cross-correlation time shifts:", cross_corr_time_shifts)
print("Optimization time shifts:", optimization_time_shifts)

mode_time_shift = -85

# Time-sync the Qualisys data based on the mode time shift
# Time-sync the Qualisys data based on the mode time shift
if mode_time_shift >= 0:
    synced_qualisys = qualisys_data[mode_time_shift:, qualisys_common_idx, :]
else:
    synced_qualisys = qualisys_data[:mode_time_shift, qualisys_common_idx, :]

# Make sure FreeMoCap and synced Qualisys data have the same number of frames
min_length = min(len(synced_qualisys), len(freemocap_data))
synced_qualisys = synced_qualisys[:min_length, :]
synced_freemocap = freemocap_data[:min_length, mediapipe_common_idx, :]

# Plot time-synced positions for common markers
for marker_idx, marker_name in enumerate(common_markers):
    for dim, dim_name in enumerate(['X', 'Y', 'Z']):
        plt.figure()
        
        # Normalize trajectories to start at 0
        qualisys_trajectory = synced_qualisys[:, marker_idx, dim] - synced_qualisys[0, marker_idx, dim]
        freemocap_trajectory = synced_freemocap[:, marker_idx, dim] - synced_freemocap[0, marker_idx, dim]
        
        # Plotting X, Y, Z trajectories for each marker
        plt.plot(qualisys_trajectory, label=f"Qualisys {marker_name} {dim_name}")
        plt.plot(freemocap_trajectory, label=f"FreeMoCap {marker_name} {dim_name}")

        plt.legend()
        plt.title(f"Time-Synced {dim_name} Position for {marker_name}")
        plt.xlabel("Frame")
        plt.ylabel(f"{dim_name} Position")
        plt.show()