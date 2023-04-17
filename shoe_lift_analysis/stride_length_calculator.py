from pathlib import Path
import numpy as np
from freemocap_utils.mediapipe_skeleton_builder import mediapipe_indices
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def load_specific_marker_data(marker_data:np.ndarray, joint_to_use:str, axis_to_use:int):
    #for axis to use, 0 = x axis, 1 = y axis, 2 = z axis
    joint_index = mediapipe_indices.index(joint_to_use)
    marker_position_3d = marker_data[:, joint_index, :]
    marker_position_1d = marker_position_3d[:,axis_to_use]
    marker_velocity_1d = np.diff(marker_position_1d, axis = 0)
    marker_velocity_1d = np.append(0,marker_velocity_1d)

    return marker_position_1d, marker_velocity_1d

def detect_zero_crossings(marker_velocity_data:np.ndarray, search_range=2, show_plot = False):
    threshold_to_ignore_next_crossing = 5
    zero_crossings = np.where(np.diff(np.sign(marker_velocity_data)))[0]

    thresholded_zero_crossings_frames = list(zero_crossings.copy())

    for count,frame in enumerate(thresholded_zero_crossings_frames):
        frames_to_filter_out = np.array(range(frame+1,frame+threshold_to_ignore_next_crossing))
        thresholded_zero_crossings_frames = np.setdiff1d(thresholded_zero_crossings_frames,frames_to_filter_out)

    zero_crossings = thresholded_zero_crossings_frames

    heel_strike_frames = []
    toe_off_frames = []

    #searches around the located zero crossing frames to find the frame that has the lowest velocity within the search range
    for frame in zero_crossings:
        start = max(0, frame - search_range)
        end = min(len(marker_velocity_data) - 1, frame + search_range)
        min_abs_velocity_index = start + np.argmin(np.abs(marker_velocity_data[start:end+1]))

        #if the velocity is negative at the original detected frame (which is right before the 0 crossing), then it's toe off (slope is positive)
        if marker_velocity_data[frame] > 0:
            heel_strike_frames.append(min_abs_velocity_index)
        else:
            toe_off_frames.append(min_abs_velocity_index)

    return heel_strike_frames, toe_off_frames


def divide_3d_data_into_steps(marker_position_3d_data: np.ndarray, event_frames: list):
    num_frames, num_markers, num_dimensions = marker_position_3d_data.shape

    def get_marker_step_data(dimension, marker, frame):
        current_event_frame = int(event_frames[frame])
        next_event_frame = int(event_frames[frame + 1])
        step_end_frame = next_event_frame - 1  # end this step right before the next heel strike/toe off
        step_frame_interval = list(range(current_event_frame, step_end_frame))

        marker_step_data = marker_position_3d_data[step_frame_interval, marker, dimension]

        if dimension == 0:
            marker_step_data -= marker_step_data[0]  # zero it out in the x direction only
        
        return marker_step_data

    dimension_step_dict = {
        dimension: {
            mediapipe_indices[marker]: {
                count: get_marker_step_data(dimension, marker, frame)
                for count, frame in enumerate(range(len(event_frames) - 1))
            }
            for marker in range(num_markers)
        }
        for dimension in range(num_dimensions)
    }

    return dimension_step_dict
                
def resample_steps(step_data_dict: dict, num_resampled_points: int):
    resampled_step_dict = {
        dimension: {
            marker: {
                step_num: resample_data(step_data, num_resampled_points=num_resampled_points)
                for step_num, step_data in step_dict.items()
            }
            for marker, step_dict in marker_dict.items()
        }
        for dimension, marker_dict in step_data_dict.items()
    }

    return resampled_step_dict

def resample_data(original_data:np.ndarray,num_resampled_points:int):
    num_current_samples = original_data.shape[0]
    end_time = 10
    original_data_time_array = np.linspace(0,end_time,num_current_samples)
    resampled_data_time_array = np.linspace(0,end_time,num_resampled_points)
    resampled_data = np.empty([resampled_data_time_array.shape[0]])

    interpolation_function = interp1d(original_data_time_array,original_data)
    resampled_data[:] = interpolation_function(resampled_data_time_array)

    return resampled_data

def calculate_stride_length(heel_data_dict: dict):
    left_heel_strike_positions = heel_data_dict['left_heel']['position'][heel_data_dict['left_heel']['heel strike']]
    right_heel_strike_positions = heel_data_dict['right_heel']['position'][heel_data_dict['right_heel']['heel strike']]

    step_lengths = []

    # Calculate step lengths for both legs
    for i in range(min(len(left_heel_strike_positions), len(right_heel_strike_positions)) - 1):
        left_step_length = abs(left_heel_strike_positions[i + 1] - left_heel_strike_positions[i])
        right_step_length = abs(right_heel_strike_positions[i + 1] - right_heel_strike_positions[i])

        step_lengths.append({"left_step_length": left_step_length, "right_step_length": right_step_length})

    return step_lengths

if __name__ == '__main__':
    #path_to_recording_folder = Path(r'C:\Users\Aaron\Documents\freemocap_sessions\recordings')
    path_to_recording_folder = Path(r'C:\Users\aaron\FreeMocap_Data\recording_sessions')


    session_id_list = ['recording_15_19_00_gmt-4__brit_baseline','recording_15_20_51_gmt-4__brit_half_inch', 'recording_15_22_56_gmt-4__brit_one_inch','recording_15_24_58_gmt-4__brit_two_inch']
    label_list = ['baseline', 'half inch lift', 'one inch lift', 'two inch lift']
    
    figure, axes = plt.subplots(2, 1)

    limb_one_axis, limb_two_axis = axes

    for session_id, label in zip(session_id_list, label_list):
        path_to_data = path_to_recording_folder/session_id/'output_data'/'mediapipe_body_3d_xyz_transformed.npy'

        marker_data_3d = np.load(path_to_data)
        marker_data_3d[:,:,0] = marker_data_3d[:,:,0]*-1

        heel_data_dict = {}

        for heel in ['left_heel', 'right_heel']:
            marker_position, marker_velocity = load_specific_marker_data(marker_data=marker_data_3d, joint_to_use= heel, axis_to_use = 0)
            heel_strike_frames, toe_off_frames = detect_zero_crossings(marker_velocity_data=marker_velocity,search_range=2)

            heel_data_dict[heel] = {}

            heel_data_dict[heel]['position'] = marker_position
            heel_data_dict[heel]['velocity'] = marker_velocity
            heel_data_dict[heel]['heel strike'] = heel_strike_frames
            heel_data_dict[heel]['toe off'] = toe_off_frames

        step_lengths = calculate_stride_length(heel_data_dict)
        f = 2


        step_data_3d = divide_3d_data_into_steps(marker_data_3d,heel_strike_frames)
        step_data_left_heel_x = step_data_3d[0]['left_heel']
        resampled_step_data_3d = resample_steps(step_data_dict=step_data_3d, num_resampled_points=100)

        # step_stats_dict = calculate_step_length_stats(step_data_3d=resampled_step_data_3d)