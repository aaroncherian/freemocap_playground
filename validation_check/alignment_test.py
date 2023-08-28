import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
from scipy import optimize
from scipy.spatial.transform import Rotation

from PyQt6.QtWidgets import QMainWindow, QWidget, QApplication, QHBoxLayout,QVBoxLayout


from rmse_viewer_gui import RMSEViewerGUI

qualisys_indices = [
'head',
'left_ear',
'right_ear',
'cspine',
'left_shoulder',
'right_shoulder',
'left_elbow',
'right_elbow',
'left_wrist',
'right_wrist',
'left_index',
'right_index',
'left_hip',
'right_hip',
'left_knee',
'right_knee',
'left_ankle',
'right_ankle',
'left_heel',
'right_heel',
'left_foot_index',
'right_foot_index',
]

mediapipe_indices = [
    'nose',
    'left_eye_inner',
    'left_eye',
    'left_eye_outer',
    'right_eye_inner',
    'right_eye',
    'right_eye_outer',
    'left_ear',
    'right_ear',
    'mouth_left',
    'mouth_right',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_pinky',
    'right_pinky',
    'left_index',
    'right_index',
    'left_thumb',
    'right_thumb',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle',
    'left_heel',
    'right_heel',
    'left_foot_index',
    'right_foot_index'
    ]



def optimize_transformation(params, segments_list_A, segments_list_B, transformed_skeletons_list):
    tx, ty, tz = params[0:3]  # Translation parameters
    rx, ry, rz = params[3:6]  # Rotation parameters (Euler angles in degrees)
    s = params[6]  # Scaling parameter

    rotation = Rotation.from_euler('xyz', [rx, ry, rz], degrees=True)
    
    total_error_list = []
    
    for segmentA, segmentB in zip(segments_list_A, segments_list_B):
        segmentA_transformed = s * rotation.apply(segmentA) + [tx, ty, tz]
        error_list = [abs(y - x) for x, y in zip(segmentA_transformed, segmentB)]
        this_segment_error = np.mean(error_list)
        transformed_skeletons_list.append(segmentA_transformed)
        total_error_list.append(this_segment_error)

    return total_error_list

def get_optimized_transformation_matrix(segments_list_A, segments_list_B):
    initial_guess = [0, 0, 0, 0, 0, 0, 1]
    transformed_skeleton_list = []

    # New optimization call
    transformation_matrix = optimize.least_squares(optimize_transformation, 
                                            initial_guess, 
                                            args=(segments_list_A, segments_list_B, transformed_skeleton_list),
                                            gtol=1e-10, 
                                            verbose=2).x
    
    return transformation_matrix, transformed_skeleton_list
    

def plot_optimization_steps(qualisys_data, transformed_skeletons_list):
    def plot_frame(f):
        ax.clear()
        ax.scatter(qualisys_data[:, 0], qualisys_data[:, 1], qualisys_data[:, 2], c='blue', label='Qualisys')
        ax.scatter(transformed_skeletons_list[f][:, 0], transformed_skeletons_list[f][:, 1], transformed_skeletons_list[f][:, 2], c='red', label='Transformed FreeMoCap')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title(f"Iteration {f}")
        fig.canvas.draw_idle()

    fig = plt.figure(figsize=[10, 8])
    ax = fig.add_subplot(111, projection='3d')
    slider_ax = plt.axes([0.25, 0.02, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    frame_slider = Slider(slider_ax, 'Iteration', 0, len(transformed_skeletons_list) - 1, valinit=0, valstep=1)

    def update(val):
        frame = int(frame_slider.val)
        plot_frame(frame)

    frame_slider.on_changed(update)
    plot_frame(0)
    plt.show()

def plot_3d_scatter(freemocap_data, qualisys_data):
    def plot_frame(f):
        ax.clear()
        ax.scatter(qualisys_data[f, :, 0], qualisys_data[f, :, 1], qualisys_data[f, :, 2], c='blue', label='Qualisys')
        ax.scatter(freemocap_data[f, :, 0], freemocap_data[f, :, 1], freemocap_data[f, :, 2], c='red', label='FreeMoCap')
        ax.set_xlim([-limit_x, limit_x])
        ax.set_ylim([-limit_y, limit_y])
        ax.set_zlim([-limit_z, limit_z])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title(f"Frame {f}")
        fig.canvas.draw_idle()

    mean_x = (np.mean(qualisys_data[:, :, 0]) + np.mean(freemocap_data[:, :, 0])) / 2
    mean_y = (np.mean(qualisys_data[:, :, 1]) + np.mean(freemocap_data[:, :, 1])) / 2
    mean_z = (np.mean(qualisys_data[:, :, 2]) + np.mean(freemocap_data[:, :, 2])) / 2

    ax_range = 1000
    limit_x = mean_x + ax_range
    limit_y = mean_y + ax_range
    limit_z = mean_z + ax_range

    fig = plt.figure(figsize=[10, 8])
    ax = fig.add_subplot(111, projection='3d')
    slider_ax = plt.axes([0.25, 0.02, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    frame_slider = Slider(slider_ax, 'Frame', 0, len(qualisys_data) - 1, valinit=0, valstep=1)

    def update(val):
        frame = int(frame_slider.val)
        plot_frame(frame)

    frame_slider.on_changed(update)
    plot_frame(0)
    plt.show()

def get_optimal_translation_matrix(segments_list_A, segments_list_B):
    translation_matrix = optimize.least_squares(get_translation_error_between_two_rotation_matrices,
                                    [0,0,0], args = (segments_list_A, segments_list_B),
                                    gtol = 1e-10,
                                    verbose = 2).x
    return translation_matrix

def get_translation_error_between_two_rotation_matrices(translation_guess,segments_list_A, segments_list_B):
    #convert euler angles to rotation matrix
    total_error_list = []
    for segmentA, segmentB in zip(segments_list_A, segments_list_B):

        segmentA_translated_by_guess = segmentA + translation_guess

        error_list = [abs(y-x) for x,y in zip(segmentA_translated_by_guess,segmentB)]

        this_segment_error = np.mean(error_list)

        total_error_list.append(this_segment_error)

    return total_error_list

def extract_corresponding_segments(data, indices, segments_definition):
    segments_list = []
    
    for segment_name, joint_names in segments_definition.items():
        segment = [data[indices.index(joint_name), :] for joint_name in joint_names]
        segments_list.append(segment)
    
    return np.array(segments_list)


segments_definition = {
    'right_upper_arm': ['right_shoulder', 'right_elbow'],
    'left_upper_arm': ['left_shoulder', 'left_elbow'],
    # 'right_forearm': ['right_elbow', 'right_wrist'],
    # 'left_forearm': ['left_elbow', 'left_wrist'],
    # 'right_hand': ['right_wrist', 'right_index'],
    # 'left_hand': ['left_wrist', 'left_index'],
    # 'right_thigh': ['right_hip', 'right_knee'],
    # 'left_thigh': ['left_hip', 'left_knee'],
    # 'right_shin': ['right_knee', 'right_ankle'],
    # 'left_shin': ['left_knee', 'left_ankle'],
    # 'right_foot': ['right_ankle', 'right_foot_index'],
    # 'left_foot': ['left_ankle', 'left_foot_index'],
}

qualisys_data_path = r"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\qualisys_MDN_NIH_Trial3\output_data\clipped_qualisys_skel_3d.npy"
freemocap_data_path = r"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_14_53_48_MDN_NIH_Trial3\output_data\mediapipe_body_3d_xyz.npy"

qualisys_data = np.load(qualisys_data_path)
freemocap_data = np.load(freemocap_data_path)
freemocap_data = freemocap_data 
# plot_3d_scatter(freemocap_data, qualisys_data)

frames_subset = [300,350]
qualisys_data_subset = qualisys_data[frames_subset[0]:frames_subset[1]]
freemocap_data_subset = freemocap_data[frames_subset[0]:frames_subset[1]]

qualisys_representative_mean = np.mean(qualisys_data_subset, axis = 0)
freemocap_representative_mean = np.mean(freemocap_data_subset, axis = 0)

freemocap_heel_midpoint_mean = np.mean([freemocap_representative_mean[mediapipe_indices.index('left_heel'), :],
                                        freemocap_representative_mean[mediapipe_indices.index('right_heel'), :]], axis=0)

qualisys_heel_midpoint_mean = np.mean([qualisys_representative_mean[qualisys_indices.index('left_heel'), :],
                                        qualisys_representative_mean[qualisys_indices.index('right_heel'), :]], axis=0)

# freemocap_data_zeroed = freemocap_data - freemocap_heel_midpoint_mean
# qualisys_data_zeroed = qualisys_data - qualisys_heel_midpoint_mean

# plot_3d_scatter(freemocap_data_zeroed, qualisys_data_zeroed)

freemocap_data_zeroed = freemocap_data
qualisys_data_zeroed = qualisys_data

freemocap_zeroed_mean = np.mean(freemocap_data_zeroed[frames_subset[0]:frames_subset[1]], axis = 0)
qualisys_zeroed_mean = np.mean(qualisys_data_zeroed[frames_subset[0]:frames_subset[1]], axis = 0)

freemocap_segmented = extract_corresponding_segments(freemocap_zeroed_mean, mediapipe_indices, segments_definition)
qualisys_segmented = extract_corresponding_segments(qualisys_zeroed_mean, qualisys_indices, segments_definition)

# transformation_matrix = get_optimal_translation_matrix(freemocap_segmented, qualisys_segmented)
transformation_matrix, transformed_skeletons_list = get_optimized_transformation_matrix(freemocap_segmented, qualisys_segmented)
plot_optimization_steps(qualisys_data = qualisys_representative_mean, transformed_skeletons_list = transformed_skeletons_list)

# translation_matrix = get_optimal_translation_matrix(freemocap_segmented, qualisys_segmented)
# freemocap_data_zeroed_translated = freemocap_data_zeroed + translation_matrix

# # freemocap_data_translated = translate_to_align(freemocap_data, qualisys_data)
# plot_3d_scatter(freemocap_data_zeroed_translated, qualisys_data)


f = 2


# app = QApplication([])
# win = RMSEViewerGUI(qualisys_data_original=qualisys_data, freemocap_data_original=freemocap_data_zeroed_translated)

# win.show()
# app.exec()