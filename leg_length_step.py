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

def calculate_step_length_stats(step_data_3d:dict):

    step_stats_dict = {
        dimension:{
            marker: {
                'mean': np.mean(list(step_dict.values()),axis = 0),
                'median': np.median(list(step_dict.values()),axis = 0),
                'std': np.std(list(step_dict.values()),axis = 0) 
            }
            for marker, step_dict in marker_dict.items() 
        }
        for dimension, marker_dict in step_data_3d.items()
    }   

    return step_stats_dict
    
# def calculate_leg_length(step_stat_dict:dict, leg_side:str):

#     lower_leg_length = calculate_distance(step_stat_dict[f'{leg_side}_knee']['mean'],step_stat_dict[f'{leg_side}_heel']['mean'])
#     upper_leg_length = calculate_distance(step_stat_dict[f'{leg_side}_hip']['mean'],step_stat_dict[f'{leg_side}_knee']['mean'])
#     leg_length = lower_leg_length + upper_leg_length
    
#     return leg_length

def calculate_leg_length(step_data_3d: dict, leg_side: str):
    n_steps = len(next(iter(step_data_3d[0].values())))
    leg_lengths = []

    for step_num in range(n_steps):
        for frame in range(100):
            lower_leg_length = calculate_distance(
                [step_data_3d[0][f"{leg_side}_knee"][step_num], step_data_3d[1][f"{leg_side}_knee"][step_num], step_data_3d[2][f"{leg_side}_knee"][step_num]],
                [step_data_3d[0][f"{leg_side}_ankle"][step_num], step_data_3d[1][f"{leg_side}_ankle"][step_num], step_data_3d[2][f"{leg_side}_ankle"][step_num]],
            )
            upper_leg_length = calculate_distance(
                [step_data_3d[0][f"{leg_side}_hip"][step_num], step_data_3d[1][f"{leg_side}_hip"][step_num], step_data_3d[2][f"{leg_side}_hip"][step_num]],
                [step_data_3d[0][f"{leg_side}_knee"][step_num], step_data_3d[1][f"{leg_side}_knee"][step_num], step_data_3d[2][f"{leg_side}_knee"][step_num]],
            )
            leg_length = lower_leg_length + upper_leg_length
            leg_lengths.append(leg_length)
            f = 2 

    return np.mean(leg_lengths)

def calculate_distance(point1, point2):
    return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2 + (point1[2]-point2[2])**2)

def create_leg_length_dict(step_stats:dict, label:str):    
    leg_length_dict = {
        label:{
            'left_leg_length': [calculate_leg_length(step_data_3d=step_stats,leg_side='left')],
            'right_leg_length': [calculate_leg_length(step_data_3d=step_stats,leg_side='right')]
        }
    }
    return leg_length_dict

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
    ax.set_xticklabels(labels)
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

def plot_leg_asymmetries_grouped_by_leg(leg_length_dicts, labels):
    n_sessions = len(leg_length_dicts)
    index = np.arange(2)
    bar_width = 0.2

    fig, ax = plt.subplots()
    colors = ['b', 'g', 'r', 'c']

    for i, (leg_length_dict, label, color) in enumerate(zip(leg_length_dicts, labels, colors)):
        left_leg_length = leg_length_dict[label]['left_leg_length'][0]
        right_leg_length = leg_length_dict[label]['right_leg_length'][0]

        rects1 = ax.bar(index[0] + i * bar_width, left_leg_length, bar_width, label=label, color=color)
        rects2 = ax.bar(index[1] + i * bar_width, right_leg_length, bar_width, color=color)

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



if __name__ == '__main__':
    #path_to_recording_folder = Path(r'C:\Users\Aaron\Documents\freemocap_sessions\recordings')
    path_to_recording_folder = Path(r'C:\Users\aaron\FreeMocap_Data\recording_sessions')


    session_id_list = ['recording_15_19_00_gmt-4__brit_baseline','recording_15_20_51_gmt-4__brit_half_inch', 'recording_15_22_56_gmt-4__brit_one_inch','recording_15_24_58_gmt-4__brit_two_inch']
    label_list = ['baseline', 'half inch lift', 'one inch lift', 'two inch lift']
    leg_length_list = []

    for session_id, label in zip(session_id_list, label_list):
        path_to_data = path_to_recording_folder/session_id/'output_data'/'mediapipe_body_3d_xyz_transformed.npy'

        marker_data_3d = np.load(path_to_data)
        marker_data_3d[:,:,0] = marker_data_3d[:,:,0]*-1

        marker_position, marker_velocity = load_specific_marker_data(marker_data=marker_data_3d, joint_to_use='left_heel', axis_to_use = 0)
        heel_strike_frames, toe_off_frames = detect_zero_crossings(marker_velocity_data=marker_velocity,search_range=2)

        step_data_3d = divide_3d_data_into_steps(marker_data_3d,heel_strike_frames)
        resampled_step_data_3d = resample_steps(step_data_dict=step_data_3d, num_resampled_points=100)
        step_stats_dict = calculate_step_length_stats(step_data_3d=resampled_step_data_3d)

        # reshape_step_data_to_frame_marker_dim(step_data_3d=resampled_step_data_3d)
        
        leg_length_list.append(create_leg_length_dict(resampled_step_data_3d, label= label))
        
    plot_leg_asymmetries(leg_length_list, label_list)
    plot_leg_asymmetries_grouped_by_leg(leg_length_list, label_list)


    f =2 
