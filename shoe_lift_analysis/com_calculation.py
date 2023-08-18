from pathlib import Path

from full_leg_marker_traces import (
    load_specific_marker_data,
    detect_zero_crossings,
    divide_3d_data_into_steps,
    resample_steps,
    calculate_step_length_stats
)

import numpy as np 
import matplotlib.pyplot as plt

def plot_com_heights(left_com_midstance_height:dict, right_com_midstance_height:dict):
    left_com_midstance_values = list(left_com_midstance_height.values())
    right_com_midstance_values = list(right_com_midstance_height.values())

    labels = label_list

    # Set up the position of the bars
    bar_width = 0.35
    left_bar_positions = np.arange(len(labels))
    right_bar_positions = left_bar_positions + bar_width

    # Create the bar graph
    fig, ax = plt.subplots()

    left_bars = ax.bar(left_bar_positions, left_com_midstance_values, bar_width, label="Left COM Height")
    right_bars = ax.bar(right_bar_positions, right_com_midstance_values, bar_width, label="Right COM Height")

    # Add labels, title, and legend
    ax.set_ylabel("COM Height (mm)")
    ax.set_title("COM Height at Midstance")
    ax.set_xticks(left_bar_positions + bar_width / 2)
    ax.set_xticklabels(labels)
    ax.legend()

    # Add values above the bars
    for bars in (left_bars, right_bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    plt.show()


if __name__ == '__main__':
    path_to_recording_folder = Path(r'D:\2023-06-07_JH\1.0_recordings\treadmill_calib')
    session_id_list = ['sesh_2023-06-07_12_38_16_JH_leg_length_neg_5_trial_1','sesh_2023-06-07_12_43_15_JH_leg_length_neg_25_trial_1', 'sesh_2023-06-07_12_46_54_JH_leg_length_neutral_trial_1','sesh_2023-06-07_12_50_56_JH_leg_length_pos_25_trial_1', 'sesh_2023-06-07_12_55_21_JH_leg_length_pos_5_trial_1']
    label_list = ['-.5', '-.25', 'neutral', '+.25', '+.5']

    left_stats_dict = {}
    right_stats_dict = {}

    left_com_midstance_height = {}
    right_com_midstance_height = {}

    for session_id, label in zip(session_id_list, label_list):
        path_to_com_data = path_to_recording_folder/session_id/'output_data'/'center_of_mass'/'total_body_center_of_mass_xyz.npy'
        com_data_3d = np.load(path_to_com_data)
        com_data_3d[:,0] = com_data_3d[:,0]*-1
        com_data_3d = np.expand_dims(com_data_3d, axis=1)

        path_to_marker_data = path_to_recording_folder/session_id/'output_data'/'mediapipe_body_3d_xyz.npy'
        marker_data_3d = np.load(path_to_marker_data)
        marker_data_3d[:,:,0] = marker_data_3d[:,:,0]*-1

        # Left heel
        marker_position_left, marker_velocity_left = load_specific_marker_data(marker_data=marker_data_3d, joint_to_use='left_heel', axis_to_use=0)
        left_heel_strike_frames, _ = detect_zero_crossings(marker_velocity_data=marker_velocity_left, search_range=2)
        
        marker_position_right, marker_velocity_right = load_specific_marker_data(marker_data=marker_data_3d,joint_to_use='right_heel',axis_to_use=0)
        right_heel_strike_frames,_ = detect_zero_crossings(marker_velocity_data=marker_velocity_right,search_range=2)

        unsampled_left_com_step_data = divide_3d_data_into_steps(com_data_3d,left_heel_strike_frames)
        left_com_step_data = resample_steps(step_data_dict=unsampled_left_com_step_data, num_resampled_points=100)
        left_com_step_stats_dict = calculate_step_length_stats(step_data_3d=left_com_step_data)
        left_stats_dict[label] = left_com_step_stats_dict

        left_com_midstance_height[label] = left_com_step_stats_dict[2]['nose']['mean'][29]

        unsampled_right_com_step_data = divide_3d_data_into_steps(com_data_3d,right_heel_strike_frames)
        right_com_step_data = resample_steps(step_data_dict=unsampled_right_com_step_data, num_resampled_points=100)
        right_com_step_stats_dict = calculate_step_length_stats(step_data_3d=right_com_step_data)
        right_stats_dict[label] = right_com_step_stats_dict

        left_com_midstance_height[label] = left_com_step_stats_dict[2]['nose']['mean'][29]
        right_com_midstance_height[label] = right_com_step_stats_dict[2]['nose']['mean'][29]

    
    plot_com_heights(left_com_midstance_height = left_com_midstance_height, right_com_midstance_height=right_com_midstance_height)
    #plot_com_height_trajectory_separate(left_stats_dict, right_stats_dict, label_list)

f = 2
# plot_leg_markers(left_session_step_stats=left_stats_dict, right_session_step_stats=right_stats_dict, dimension_to_plot=2, labels=label_list)
