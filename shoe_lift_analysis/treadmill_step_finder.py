from pathlib import Path
import numpy as np
from freemocap_utils.mediapipe_skeleton_builder import mediapipe_indices
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d





# freemocap_all_marker_data = np.load(path_to_data)
# joint_index = mediapipe_indices.index(joint_to_use)
# direction = 0

# freemocap_marker_position = freemocap_all_marker_data[:,joint_index,direction] *-1
# freemocap_marker_velocity = np.diff(freemocap_marker_position,axis = 0)

def load_marker_position_and_velocity(path_to_data:Path, joint_to_use:str, axis_to_use:int):
    marker_data = np.load(path_to_data)
    marker_data = marker_data[:,:,:]
    
    marker_data[:,:,0] = marker_data[:,:,0] *-1

    marker_position, marker_velocity = load_specific_marker_data(marker_data=marker_data, joint_to_use=joint_to_use,axis_to_use = axis_to_use)

    return marker_position,marker_velocity

def load_specific_marker_data(marker_data:np.ndarray, joint_to_use:str, axis_to_use:int):
    #for axis to use, 0 = x axis, 1 = y axis, 2 = z axis
    joint_index = mediapipe_indices.index(joint_to_use)
    marker_position_3d = marker_data[:, joint_index, :]
    marker_position_1d = marker_position_3d[:,axis_to_use]

    num_frames = marker_position_1d.shape[0]

    # for frame in range(num_frames):
    #     marker_position_1d[frame] = marker_position_1d[frame] + 40 * frame

    marker_velocity_1d = np.diff(marker_position_1d, axis = 0)
    marker_velocity_1d = np.append(0,marker_velocity_1d)

    return marker_position_1d, marker_velocity_1d

 #x = 0,y = 1, z = 2

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

        #to account for the np diff removes the first frame
        # heel_strike_frames_adjusted = [frame + 1 for frame in heel_strike_frames]
        # toe_off_frames_adjusted = [frame + 1 for frame in toe_off_frames]

    return heel_strike_frames, toe_off_frames

def plot_event_frames(marker_position_data:np.ndarray, marker_velocity_data:np.ndarray, heel_strike_frames, toe_off_frames):
    figure = plt.figure()
    position_ax = figure.add_subplot(211)
    velocity_ax = figure.add_subplot(212)

    position_ax.set_title(f'{joint_to_use} position vs. frame')
    velocity_ax.set_title(f'{joint_to_use} velocity vs. frame')

    position_ax.set_ylabel('Joint Position (mm)')
    velocity_ax.set_ylabel('Joint Velocity (mm)')
    velocity_ax.set_xlabel('Frame #')


    velocity_ax.axhline(0, alpha = .6, color = 'k', linestyle = '--')

    position_ax.plot(marker_position_data, '.-', color = 'k', alpha = .6)
    velocity_ax.plot(marker_velocity_data, '.-', color = 'k', alpha = .6)

    position_ax.scatter(toe_off_frames,marker_position_data[toe_off_frames], marker = 'o',color = 'r', label = 'toe off')
    position_ax.scatter(heel_strike_frames,marker_position_data[heel_strike_frames], marker = 'o',color = 'b', label = 'heel strike')

    velocity_ax.scatter(toe_off_frames,marker_velocity_data[toe_off_frames], marker = 'o',color = 'r', label = 'toe off')
    velocity_ax.scatter(heel_strike_frames,marker_velocity_data[heel_strike_frames], marker = 'o',color = 'b', label = 'heel strike')
    velocity_ax.legend()

    plt.show()


def resample_step_data_dict(step_data_dict:dict, num_resampled_points:int):
    resampled_step_data_dict = {}

    for step_num in step_data_dict.keys():
        resampled_step_data, resampled_data_time_array, original_data_time_array = resample_data(step_data_dict[step_num],num_resampled_points=num_resampled_points)
        resampled_step_data_dict[step_num] = resampled_step_data

    return resampled_step_data_dict


def resample_data(original_data:np.ndarray,num_resampled_points:int):
    num_current_samples = original_data.shape[0]
    end_time = 10
    original_data_time_array = np.linspace(0,end_time,num_current_samples)
    resampled_data_time_array = np.linspace(0,end_time,num_resampled_points)
    resampled_data = np.empty([resampled_data_time_array.shape[0]])

    interpolation_function = interp1d(original_data_time_array,original_data)
    resampled_data[:] = interpolation_function(resampled_data_time_array)

    return resampled_data, resampled_data_time_array, original_data_time_array
    f = 2

def calculate_step_lengths(marker_position_data:np.ndarray, event_frames:list):
    #get the step data for a heel strike/toe off - from the start of the first event to one frame before the next
    step_data_dict = {}

    for count,frame in enumerate(range(len(event_frames)-1)):
        current_event_frame = int(event_frames[frame])
        next_event_frame = int(event_frames[frame+1]) 
        step_end_frame = next_event_frame - 1
        step_frames = list(range(current_event_frame,step_end_frame))
        step_data = marker_position_data[step_frames]
        # step_data = marker_position_data[step_frames] - marker_position[step_frames][0] #zero it out
        step_data_dict[count] = step_data

    return step_data_dict

def calculate_step_trajectory_stats(step_data_dict:dict):
    step_data_mean = np.mean(list(step_data_dict.values()),axis = 0)
    step_data_std = np.std(list(step_data_dict.values()),axis = 0)    
    step_data_median = np.median(list(step_data_dict.values()),axis = 0)

    return step_data_mean, step_data_median,step_data_std

def plot_avg_step_trajectory(step_data_mean:np.ndarray,step_data_median:np.ndarray, step_data_std:np.ndarray,step_data_dict:dict):

    figure = plt.figure()
    position_ax = figure.add_subplot(111)
    position_ax.set_title(f'{joint_to_use} Average Step Trajectory')
    position_ax.set_ylabel('X (Forward) Position (mm)')
    position_ax.set_xlabel('Frame #')

    x = np.arange(len(step_data_mean))

    for step_num in step_data_dict.keys():
        position_ax.plot(x,step_data_dict[step_num], alpha = .3, color = 'grey')

    position_ax.plot(x,step_data_mean, color = 'k', label = 'mean')
    position_ax.fill_between(x,step_data_mean-step_data_std, step_data_mean + step_data_std, color = 'g', alpha = .2)
    
    position_ax.plot(x,step_data_median, color = 'k', linestyle ='--', label= 'median')
    position_ax.legend()
    plt.show()

def plot_all_recording_means(recordings_step_mean_dict:dict, joint_to_use):
    figure = plt.figure()
    position_ax = figure.add_subplot(111)
    position_ax.set_title(f'{joint_to_use} Average Step Trajectory')
    position_ax.set_ylabel('X (Forward) Position (mm)')
    position_ax.set_xlabel('Frame #')

    for recording_name in recordings_step_mean_dict.keys():
        position_ax.plot(recordings_step_mean_dict[recording_name], label = recording_name)

    position_ax.legend()
    plt.show()

def calculate_average_step_length(marker_position, event_frames):
    step_data_dict = calculate_step_lengths(marker_position_data=marker_position, event_frames=event_frames)
    resampled_data_dict = resample_step_data_dict(step_data_dict=step_data_dict, num_resampled_points=100)
    step_data_mean, step_data_median, step_data_std = calculate_step_trajectory_stats(step_data_dict=resampled_data_dict)
    return step_data_mean, step_data_median, step_data_std, resampled_data_dict

def plot_x_vs_z(x_mean_step:np.ndarray,z_mean_step:np.ndarray):
    figure = plt.figure()
    position_ax = figure.add_subplot(111)
    position_ax.set_title(f'{joint_to_use} Average Step Trajectory')
    position_ax.set_ylabel('Z Position (mm)')
    position_ax.set_xlabel('X Position (mm)')

    position_ax.plot(x_mean_step,z_mean_step)
    
    num_frames = len(x_mean_step)
    midpoint_frame = int(num_frames/2)
    
    position_ax.scatter(x_mean_step[0],z_mean_step[0], color = 'b', marker = 'p')
    position_ax.scatter(x_mean_step[-1],z_mean_step[-1], color = 'r', marker = 'o')
    position_ax.scatter(x_mean_step[midpoint_frame],z_mean_step[midpoint_frame], color = 'm', marker = 'h')
    # plt.show()

def plot_all_saggittal(x_dict,z_dict):
    
    figure = plt.figure()
    position_ax = figure.add_subplot(111)
    position_ax.set_title(f'{joint_to_use} Average Step Trajectory')
    position_ax.set_ylabel('Z Position (mm)')
    position_ax.set_xlabel('X Position (mm)')

    for x_mean_key in x_dict.keys():
        position_ax.plot(x_dict[x_mean_key],z_dict[x_mean_key], label = x_mean_key)
    position_ax.legend()
    
    # num_frames = len(x_mean_step)
    # midpoint_frame = int(num_frames/2)
    
    # position_ax.scatter(x_mean_step[0],z_mean_step[0], color = 'b', marker = 'p')
    # position_ax.scatter(x_mean_step[-1],z_mean_step[-1], color = 'r', marker = 'o')
    # position_ax.scatter(x_mean_step[midpoint_frame],z_mean_step[midpoint_frame], color = 'm', marker = 'h')
    # plt.show()

def calculate_rmse(baseline_data: np.ndarray, comparison_data: np.ndarray):
    if baseline_data.shape != comparison_data.shape:
        raise ValueError("Baseline and comparison data must have the same shape")

    
    squared_diff = np.abs((baseline_data - comparison_data)) 
    mean_squared_diff = np.mean(squared_diff)
    return mean_squared_diff

def compare_sessions_rmse(session_id_list, label_list, joint_to_use,step_data_mean_dict):
    rmse_dict = {}
    baseline_label = label_list[0]
    baseline_data = step_data_mean_dict[baseline_label]

    for session_id, label in zip(session_id_list[1:], label_list[1:]):
        comparison_data = step_data_mean_dict[label]
        rmse = calculate_rmse(baseline_data[60:100], comparison_data)
        rmse_dict[label] = rmse

    return rmse_dict
    
def calculate_everything_for_joint(path_to_data, joint_to_use, axis_to_use, event_frames):
    marker_position, marker_velocity = load_marker_position_and_velocity(path_to_data=path_to_data, joint_to_use=joint_to_use, axis_to_use = axis_to_use)
    step_data_mean, step_data_median, step_data_std, resampled_step_dict = calculate_average_step_length(marker_position=marker_position,event_frames=event_frames)

    return step_data_mean, resampled_step_dict


if __name__ == '__main__':
    #path_to_recording_folder = Path(r'C:\Users\Aaron\Documents\freemocap_sessions\recordings')
    path_to_recording_folder = Path(r'D:\2023-06-07_JH\1.0_recordings\treadmill_calib')
    session_id_list = ['sesh_2023-06-07_12_38_16_JH_leg_length_neg_5_trial_1']
    label_list = ['-.5']
    
    

    # session_id_list = ['recording_15_19_00_gmt-4__brit_baseline']
    # label_list = ['baseline']dsddaavd
    joint_to_use = 'left_heel'
    step_data_mean_dict = {}
    step_data_std_dict = {}
    step_data_median_dict = {}

    step_data_mean_dict_z = {}
    step_data_mean_right_dict = {}
    # full_two_inch_data = np.load(path_to_recording_folder/session_id_list[3]/'output_data'/'full_mediapipe_body_3d_xyz_transformed.npy')
    # clipped_data = full_two_inch_data[215:1000,:,:]
    # np.save(path_to_recording_folder/session_id_list[3]/'output_data'/'mediapipe_body_3d_xyz_transformed.npy', clipped_data)




    for session_id, label in zip(session_id_list, label_list):
        path_to_data = path_to_recording_folder/session_id/'output_data'/'mediapipe_body_3d_xyz.npy'

        
        marker_position, marker_velocity = load_marker_position_and_velocity(path_to_data=path_to_data, joint_to_use=joint_to_use, axis_to_use = 0)
        heel_strike_frames, toe_off_frames = detect_zero_crossings(marker_velocity_data=marker_velocity,search_range=2)
        
        step_data_mean, step_data_median, step_data_std, resampled_step_dict = calculate_average_step_length(marker_position=marker_position,event_frames=heel_strike_frames)
   
        step_data_mean_dict[label] = step_data_mean
        step_data_std_dict[label] = step_data_std

        # step_data_mean_z, step_data_std_z = calculate_average_step_length(marker_data=freemocap_data,joint_to_use=joint_to_use,axis_to_use=2,event_frames=heel_strike_frames)

        plot_event_frames(marker_position_data=marker_position, marker_velocity_data=marker_velocity, heel_strike_frames=heel_strike_frames, toe_off_frames=toe_off_frames)
        # plot_avg_step_trajectory(step_data_mean=step_data_mean, 
        #                          step_data_median = step_data_median, 
        #                          step_data_std= step_data_std,
        #                          step_data_dict=resampled_step_dict)
        
        marker_position_z, marker_velocity_z = load_marker_position_and_velocity(path_to_data=path_to_data, joint_to_use=joint_to_use, axis_to_use = 2)
        step_data_mean_z, step_data_median_z, step_data_std_z, resampled_step_dict_z = calculate_average_step_length(marker_position=marker_position_z,event_frames=heel_strike_frames)

        # marker_position_x_dict[label] = marker_position
        # marker_position_z_dict[label] = marker_position_z
        
        marker_postion_right, marker_velocity_right = load_marker_position_and_velocity(path_to_data=path_to_data, joint_to_use='right_heel', axis_to_use = 2)
        step_data_mean_right, step_data_median_right, step_data_std_right, resampled_step_dict_right = calculate_average_step_length(marker_position=marker_postion_right,event_frames=heel_strike_frames)



        step_data_mean_dict_z[label] = step_data_mean_z

        

        # plot_x_vs_z(step_data_mean,step_data_mean_z)
        # plot_x_vs_z(step_data_mean_x_right,step_data_mean_z_right)

        left_ankle_step_mean, left_ankle_step_dict = calculate_everything_for_joint(path_to_data=path_to_data,joint_to_use='left_heel', axis_to_use=2, event_frames=heel_strike_frames)
        step_data_mean_dict[label] = left_ankle_step_mean
        f = 2
            
    # plot_all_saggittal(step_data_mean_dict, step_data_mean_dict_z)
    # plot_all_recording_means(recordings_step_mean_dict=step_data_mean_dict, joint_to_use='left_heel')
    # plot_all_recording_means(recordings_step_mean_dict=step_data_mean_right_dict, joint_to_use='right_heel')
    plot_all_recording_means(recordings_step_mean_dict=step_data_mean_dict, joint_to_use='left_heel')
    plt.show()

    # rmse_results = compare_sessions_rmse(session_id_list, label_list, 'joint_to_use',step_data_mean_dict)
    # print("RMSE between baseline and other sessions:", rmse_results)
    # f = 2