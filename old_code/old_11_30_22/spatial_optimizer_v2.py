import numpy as np
from fmc_validation_toolbox.mediapipe_skeleton_builder import mediapipe_indices, build_mediapipe_skeleton, slice_mediapipe_data
from fmc_validation_toolbox.qualisys_skeleton_builder import qualisys_indices, build_qualisys_skeleton

from scipy import optimize
from scipy.spatial.transform import Rotation

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

points_to_extract = [
    'right_shoulder',
    'right_elbow',
    'right_wrist',
    'right_hip',
    'right_knee',
    'right_ankle',
    'left_shoulder',
    'left_elbow',
    'left_wrist',
    'left_hip',
    'left_knee',
    'left_ankle',
    'left_heel',
    'right_heel',
    'right_foot_index',
    'left_foot_index',
]

def extract_specific_markers(data, indices, markers_to_extract):
    """
    Extracts specific markers from a 3D data array based on the given indices and markers to extract.

    Parameters:
    - data (numpy.ndarray): The 3D data array containing all markers. Shape should be (num_frames, num_markers, 3).
    - indices (list): The list of marker names corresponding to the columns in the data array.
    - markers_to_extract (list): The list of marker names to extract.

    Returns:
    - extracted_data (numpy.ndarray): A new 3D data array containing only the extracted markers. 
      Shape will be (num_frames, num_extracted_markers, 3).
    """
    # Identify the column indices that correspond to the markers to extract
    col_indices = [indices.index(marker) for marker in markers_to_extract if marker in indices]
    
    # Extract the relevant columns from the data array
    extracted_data = data[:, col_indices, :]
    
    return extracted_data

def optimize_transformation(params, freemocap_data, qualisys_data):
    tx, ty, tz = params[0:3]  # Translation parameters
    rx, ry, rz = params[3:6]  # Rotation parameters (Euler angles in degrees)
    s = params[6]  # Scaling parameter
    
    rotation = Rotation.from_euler('xyz', [rx, ry, rz], degrees=True)
    transformed_data = s * rotation.apply(freemocap_data) + [tx, ty, tz]
    
    error = np.linalg.norm(transformed_data - qualisys_data)
    return error

def objective_least_squares(params, freemocap_data, qualisys_data):
    tx, ty, tz = params[0:3]  # Translation parameters
    rx, ry, rz = params[3:6]  # Rotation parameters (Euler angles in degrees)
    s = params[6]  # Scaling parameter
    
    rotation = Rotation.from_euler('xyz', [rx, ry, rz], degrees=True)
    transformed_freemocap_data = s * rotation.apply(freemocap_data) + [tx, ty, tz]
    
    # Calculate the residuals (errors)
    residuals = qualisys_data - transformed_freemocap_data
    return residuals.flatten()

def plot_representative_means(freemocap_data, qualisys_data):
    def plot_data():
        ax.clear()
        ax.scatter(qualisys_data[:, 0], qualisys_data[:, 1], qualisys_data[:, 2], c='blue', label='Qualisys')
        ax.scatter(freemocap_data[:, 0], freemocap_data[:, 1], freemocap_data[:, 2], c='red', label='FreeMoCap')
        ax.set_xlim([-limit_x, limit_x])
        ax.set_ylim([-limit_y, limit_y])
        ax.set_zlim([-limit_z, limit_z])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title(f"Representative means")
        fig.canvas.draw_idle()

    fig = plt.figure(figsize=[10, 8])
    ax = fig.add_subplot(111, projection='3d')

    mean_x = (np.mean(qualisys_data[:, 0]) + np.mean(freemocap_data[:, 0])) / 2
    mean_y = (np.mean(qualisys_data[:, 1]) + np.mean(freemocap_data[:, 1])) / 2
    mean_z = (np.mean(qualisys_data[:, 2]) + np.mean(freemocap_data[:, 2])) / 2

    
    ax_range = 1000
    limit_x = mean_x + ax_range
    limit_y = mean_y + ax_range
    limit_z = mean_z + ax_range

    plot_data()

    plt.show()


qualisys_data_path = r"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\qualisys_MDN_NIH_Trial3\output_data\clipped_qualisys_skel_3d.npy"
freemocap_data_path = r"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_14_53_48_MDN_NIH_Trial3\output_data\mediapipe_body_3d_xyz.npy"

qualisys_data = np.load(qualisys_data_path)
freemocap_data = np.load(freemocap_data_path)


qualisys_extracted = extract_specific_markers(qualisys_data, qualisys_indices, points_to_extract)
freemocap_extracted = extract_specific_markers(freemocap_data, mediapipe_indices, points_to_extract)

frames_subset = [300,301]

qualisys_extracted_subset = qualisys_extracted[frames_subset, :, :]
freemocap_extracted_subset = freemocap_extracted[frames_subset, :, :]

qualisys_representative_mean = np.mean(qualisys_extracted_subset, axis=0)
freemocap_representative_mean = np.mean(freemocap_extracted_subset, axis=0)

# Initial guess for the optimization
initial_guess = [0, 0, 0, 0, 0, 0, 1]  # [tx, ty, tz, rx, ry, rz, s]

# Run the optimization
minimized_params = optimize.minimize(optimize_transformation, 
                           initial_guess, 
                           args=(freemocap_representative_mean, qualisys_representative_mean),
                           method='L-BFGS-B',).x

# Extract the optimized transformation parameters

plot_representative_means(freemocap_representative_mean,qualisys_representative_mean)

tx, ty, tz = minimized_params[0:3]  # Translation parameters
rx, ry, rz = minimized_params[3:6]  # Rotation parameters (Euler angles in degrees)
s = minimized_params[6]  # Scaling parameter

rotation = Rotation.from_euler('xyz', [rx, ry, rz], degrees=True)
minimized_data = s * rotation.apply(freemocap_representative_mean) + [tx, ty, tz]

plot_representative_means(minimized_data,qualisys_representative_mean)

optimized_params = optimize.least_squares(objective_least_squares, initial_guess, args=(freemocap_representative_mean, qualisys_representative_mean),  gtol=1e-10, 
                                            verbose=2).x

tx, ty, tz = optimized_params[0:3]  # Translation parameters
rx, ry, rz = optimized_params[3:6]  # Rotation parameters (Euler angles in degrees)
s = optimized_params[6]  # Scaling parameter

rotation = Rotation.from_euler('xyz', [rx, ry, rz], degrees=True)
optimized_data = s * rotation.apply(freemocap_representative_mean) + [tx, ty, tz]

plot_representative_means(optimized_data,qualisys_representative_mean)



f = 2