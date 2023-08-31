from scipy import optimize
from scipy.spatial.transform import Rotation
import numpy as np

import numpy as np
from fmc_validation_toolbox.mediapipe_skeleton_builder import mediapipe_indices, build_mediapipe_skeleton, slice_mediapipe_data
from fmc_validation_toolbox.qualisys_skeleton_builder import qualisys_indices, build_qualisys_skeleton

from scipy import optimize
from scipy.spatial.transform import Rotation

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def optimize_transformation_least_squares(transformation_matrix_guess, data_to_transform, reference_data):
    """
    Objective function for least squares optimization to find the best 3D transformation parameters.
    The transformation matrix contains [3 translation, 3 rotation, 1 scaling].

    Parameters:
    - transformation_matrix_guess (list): Initial guess for [tx, ty, tz, rx, ry, rz, s].
    - data_to_transform (numpy.ndarray): 3D array of shape (num_frames, num_markers, 3) to be transformed.
    - reference_data (numpy.ndarray): 3D array of shape (num_frames, num_markers, 3) used as reference.

    Returns:
    - residuals (numpy.ndarray): Flattened array of residuals (differences) between transformed and reference data.
    """
    tx, ty, tz, rx, ry, rz, s = transformation_matrix_guess
    rotation = Rotation.from_euler('xyz', [rx, ry, rz], degrees=True)
    transformed_data = s * rotation.apply(data_to_transform) + [tx, ty, tz]
    residuals = reference_data - transformed_data
    return residuals.flatten()

def optimize_transformation_minimize(transformation_matrix_guess, data_to_transform, reference_data):
    """
    Objective function for minimization to find the best 3D transformation parameters.
    The transformation matrix contains [3 translation, 3 rotation, 1 scaling].

    Parameters:
    - transformation_matrix_guess (list): Initial guess for [tx, ty, tz, rx, ry, rz, s].
    - data_to_transform (numpy.ndarray): 3D array of shape (num_frames, num_markers, 3) to be transformed.
    - reference_data (numpy.ndarray): 3D array of shape (num_frames, num_markers, 3) used as reference.

    Returns:
    - error (float): Norm of the residuals (differences) between transformed and reference data.
    """
    tx, ty, tz, rx, ry, rz, s = transformation_matrix_guess
    rotation = Rotation.from_euler('xyz', [rx, ry, rz], degrees=True)
    transformed_data = s * rotation.apply(data_to_transform) + [tx, ty, tz]
    error = np.linalg.norm(transformed_data - reference_data)
    return error

def apply_transformation(transformation_matrix, data_to_transform):
    """
    Apply 3D transformation to a given dataset.
    The transformation matrix contains [3 translation, 3 rotation, 1 scaling].

    Parameters:
    - transformation_matrix (list): Transformation parameters [tx, ty, tz, rx, ry, rz, s].
    - data_to_transform (numpy.ndarray): 3D array of shape (num_frames, num_markers, 3) to be transformed.

    Returns:
    - transformed_data (numpy.ndarray): 3D array of the transformed data.
    """
    tx, ty, tz, rx, ry, rz, s = transformation_matrix
    rotation = Rotation.from_euler('xyz', [rx, ry, rz], degrees=True)
    transformed_data = np.zeros_like(data_to_transform)
    
    for i in range(data_to_transform.shape[0]):
        transformed_data[i, :, :] = s * rotation.apply(data_to_transform[i, :, :]) + [tx, ty, tz]
        
    return transformed_data

if __name__ == '__main__':
    qualisys_data_path = r"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\qualisys_MDN_NIH_Trial3\output_data\clipped_qualisys_skel_3d.npy"
    freemocap_data_path = r"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_14_53_48_MDN_NIH_Trial3\output_data\mediapipe_body_3d_xyz.npy"

    qualisys_data = np.load(qualisys_data_path)
    freemocap_data = np.load(freemocap_data_path)
