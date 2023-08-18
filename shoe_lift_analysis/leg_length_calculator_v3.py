from pathlib import Path
import numpy as np
from freemocap_utils.mediapipe_skeleton_builder import mediapipe_indices
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def calculate_distance(point1, point2):
    return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2 + (point1[2]-point2[2])**2)

def calculate_leg_length(skeleton_for_this_frame, leg_indices:list):
    upper_leg_length = calculate_distance(skeleton_for_this_frame[leg_indices[0]],skeleton_for_this_frame[leg_indices[1]])
    lower_leg_length = calculate_distance(skeleton_for_this_frame[leg_indices[1]],skeleton_for_this_frame[leg_indices[2]])
    leg_length = lower_leg_length + upper_leg_length
    return leg_length

def plot_leg_asymmetries(leg_length_dicts, labels):
    n_sessions = len(leg_length_dicts)
    index = np.arange(n_sessions)
    bar_width = 0.35

    left_leg_lengths = [d[label]['left_leg_length'][0] for d, label in zip(leg_length_dicts, labels)]
    right_leg_lengths = [d[label]['right_leg_length'][0] for d, label in zip(leg_length_dicts, labels)]

    fig, ax = plt.subplots()
    rects1 = ax.bar(index - bar_width / 2, left_leg_lengths, bar_width, label='Left Leg')
    rects2 = ax.bar(index + bar_width / 2, right_leg_lengths, bar_width, label='Right Leg')

    ax.set_ylabel('Leg Length (mm)')
    ax.set_title('Average Leg Length')
    ax.set_xticks(index)
    # ax.set_xticklabels(labels)
    ax.legend()

    # Add bar values on top
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    fig.tight_layout()
    plt.show()


def plot_leg_asymmetries_grouped_by_leg(leg_length_dicts, leg_lengths_std, labels):
    n_sessions = len(leg_length_dicts)
    index = np.arange(2)
    bar_width = 0.2

    fig, ax = plt.subplots()
    # ax.set_ylim([575,620])
    colors = ['b', 'g', 'r', 'c']
    


    for i, (leg_length_dict, label, color) in enumerate(zip(leg_length_dicts, labels, colors)):
        left_leg_length = leg_length_dicts[label]['left_leg_mean_length']
        right_leg_length = leg_length_dicts[label]['right_leg_mean_length']
        left_leg_err = leg_lengths_std[label]['left_leg_std']
        right_leg_err = leg_lengths_std[label]['right_leg_std']

        rects1 = ax.bar(index[0] + i * bar_width, left_leg_length, bar_width, yerr= left_leg_err, label=label, color=color)
        rects2 = ax.bar(index[1] + i * bar_width, right_leg_length, bar_width, yerr = right_leg_err,color=color)

        # Add bar values on top
        for rects in [rects1, rects2]:
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
    ax.set_ylabel('Leg Length')
    ax.set_title('Leg Lengths by Leg (Grouped by Left and Right)')
    ax.set_xticks(index + (n_sessions - 1) * bar_width / 2)
    ax.set_xticklabels(['Left Leg', 'Right Leg'])


    ax.legend()

    fig.tight_layout()
    plt.show()

def reshape_step_data_to_frame_marker_dim(step_data_3d: dict):
    num_dimensions = len(step_data_3d)
    num_markers = len(step_data_3d[0])
    num_frames = 100

    reshaped_data = np.empty((num_frames, num_markers, num_dimensions))

    for dimension in range(num_dimensions):
        for marker, step_data in step_data_3d[dimension].items():
            marker_index = mediapipe_indices.index(marker)
            for frame in range(num_frames):
                reshaped_data[frame, marker_index, dimension] = step_data[frame]

    return reshaped_data

def calculate_mean_leg_lengths(leg_length_dict):
    mean_leg_lengths = {}
    for label, data in leg_length_dict.items():
        mean_leg_lengths[label] = {
            'left_leg_mean_length': np.mean(data['left_leg_length']),
            'right_leg_mean_length': np.mean(data['right_leg_length'])
        }
    return mean_leg_lengths

def calculate_std_leg_lengths(leg_length_dict):
    std_leg_lengths = {}
    for label, data in leg_length_dict.items():
        std_leg_lengths[label] = {
            'left_leg_std': np.std(data['left_leg_length']),
            'right_leg_std': np.std(data['right_leg_length'])
        }
    return std_leg_lengths

if __name__ == '__main__':
    #path_to_recording_folder = Path(r'C:\Users\Aaron\Documents\freemocap_sessions\recordings')
    path_to_recording_folder = Path(r'D:\2023-06-07_JH\1.0_recordings\treadmill_calib')
    session_id_list = ['sesh_2023-06-07_12_38_16_JH_leg_length_neg_5_trial_1','sesh_2023-06-07_12_43_15_JH_leg_length_neg_25_trial_1', 'sesh_2023-06-07_12_46_54_JH_leg_length_neutral_trial_1','sesh_2023-06-07_12_50_56_JH_leg_length_pos_25_trial_1', 'sesh_2023-06-07_12_55_21_JH_leg_length_pos_5_trial_1']
    label_list = ['-.5', '-.25', 'neutral', '+.25', '+.5']
    leg_length_list = []

    left_leg_indices = [mediapipe_indices.index('left_hip'),mediapipe_indices.index('left_knee'),mediapipe_indices.index('left_ankle')]
    right_leg_indices = [mediapipe_indices.index('right_hip'),mediapipe_indices.index('right_knee'),mediapipe_indices.index('right_ankle')]

    leg_length_dict = {}

    for session_id, label in zip(session_id_list, label_list):
        path_to_data = path_to_recording_folder/session_id/'output_data'/'mediapipe_body_3d_xyz.npy'

        marker_data_3d = np.load(path_to_data)
        marker_data_3d[:,:,0] = marker_data_3d[:,:,0]*-1

        num_frames = marker_data_3d.shape[0]

        left_leg_length_list = []
        right_leg_length_list = []
        #leg_length_dict = {}

        for frame in range(num_frames):
            left_leg_length = calculate_leg_length(marker_data_3d[frame], left_leg_indices)
            right_leg_length = calculate_leg_length(marker_data_3d[frame], right_leg_indices)

            left_leg_length_list.append(left_leg_length)
            right_leg_length_list.append(right_leg_length)


        leg_length_dict[label] = {'left_leg_length': left_leg_length_list, 'right_leg_length': right_leg_length_list}
        # leg_length_list.append(leg_length_dict)

    mean_leg_length = calculate_mean_leg_lengths(leg_length_dict=leg_length_dict)
    std_leg_length = calculate_std_leg_lengths(leg_length_dict=leg_length_dict)
    # plot_leg_asymmetries(leg_length_dicts=leg_length_list, labels=label_list)
    plot_leg_asymmetries_grouped_by_leg(leg_length_dicts=mean_leg_length, leg_lengths_std = std_leg_length, labels=label_list)   
    f = 2


