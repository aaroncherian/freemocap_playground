from distutils.log import debug
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path 
import socket
import pickle
from rich.progress import track
import cv2
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import matplotlib.ticker as mticker
import sys 
from datetime import datetime
from scipy.optimize import minimize
import scipy.io as sio

#from fmc_validation_toolbox import mediapipe_skeleton_builder, qualisys_skeleton_builder

from skeleton_data_holder import SkeletonDataHolder

#mediapipe_indices = mediapipe_skeleton_builder.mediapipe_indices
#qualisys_indices = qualisys_skeleton_builder.qualisys_indices

#import debugging_plot_tools


def find_velocity_values_within_limit(skeleton_velocity_data, velocity_limit):
    """
    This function takes in a skeleton velocity data array and a limit and returns the indices of the values that are within the limit
    """
    indices = []
    for i in range(len(skeleton_velocity_data)):
        if abs(skeleton_velocity_data[i]) <= velocity_limit:
            indices.append(i+1) #add 1 to account for the difference in indices between the position and velocity data
    return indices

def find_matching_indices_in_lists(list_1,list_2,list_3,list_4):
    """
    This function takes in four lists and returns the indices of the values that are in all four lists
    """
    matching_values = [x for x in list_1 if x in list_2 and x in list_3 and x in list_4]

    return matching_values


def set_axes_ranges(plot_ax,skeleton_data,ax_range):

    mx = np.nanmean(skeleton_data[:,0])
    my = np.nanmean(skeleton_data[:,1])
    mz = np.nanmean(skeleton_data[:,2])

    plot_ax.set_xlim(mx-ax_range,mx+ax_range)
    plot_ax.set_ylim(my-ax_range,my+ax_range)
    plot_ax.set_zlim(mz-ax_range,mz+ax_range)     

def find_best_velocity_guess(skeleton_type_to_use, skeleton_velocity_data, velocity_guess, iteration_range):
    """
    This function iterates over velocity data and tries to pare down to a single frame that has the closest velocity to 0 for all foot markers
    """
    if skeleton_type_to_use == 'mediapipe':
        #skeleton_data_path = this_freemocap_data_path/'mediapipe_origin_corrected_and_rotated.npy'
        right_heel_index = 30
        right_toe_index = 32
        left_heel_index = 29
        left_toe_index = 31

    elif skeleton_type_to_use == 'qualisys':

        right_heel_index = 12
        right_toe_index = 14
        left_heel_index = 11
        left_toe_index = 13

  


    skeleton_data_velocity_x_right_heel = skeleton_velocity_data[:,right_heel_index,0]
    skeleton_data_velocity_x_right_toe = skeleton_velocity_data[:,right_toe_index,0]
    skeleton_data_velocity_x_left_heel = skeleton_velocity_data[:,left_heel_index,0]
    skeleton_data_velocity_x_left_toe = skeleton_velocity_data[:,left_toe_index,0]

    #get a list of the frames where the velocity for that marker is within the velocity limit 
    right_heel_x_velocity_limits = find_velocity_values_within_limit(skeleton_data_velocity_x_right_heel, velocity_guess)
    right_toe_x_velocity_limits = find_velocity_values_within_limit(skeleton_data_velocity_x_right_toe, velocity_guess)
    left_heel_x_velocity_limits = find_velocity_values_within_limit(skeleton_data_velocity_x_left_heel, velocity_guess)
    left_toe_x_velocity_limits = find_velocity_values_within_limit(skeleton_data_velocity_x_left_toe, velocity_guess)

    #return a list of matching frame indices from the four lists generated above 
    matching_values = find_matching_indices_in_lists(right_heel_x_velocity_limits, right_toe_x_velocity_limits, left_heel_x_velocity_limits, left_toe_x_velocity_limits)
    matching_values = [x for x in matching_values if x>75]
    
    print(matching_values)
    if len(matching_values) > 1 and velocity_guess > 0:
        #if there are multiple matching values, decrease the guess a little bit and run the function again
        #  
        velocity_guess = velocity_guess - iteration_range
        print('Current Velocity Guess: ',velocity_guess, '\n','Number of Matching Values: ', len(matching_values))
        matching_values, velocity_guess = find_best_velocity_guess(skeleton_type_to_use,skeleton_velocity_data, velocity_guess, iteration_range)

        f = 2
    elif len(matching_values) == 0:
        #if there are no matching values (we decreased our guess too far), reset the guess to be a bit smaller and run the function again with smaller intervals between the guesses
        iteration_range = iteration_range/2
        matching_values, velocity_guess = find_best_velocity_guess(skeleton_type_to_use, skeleton_velocity_data, velocity_guess + iteration_range*2, iteration_range)

        f = 2
    elif len(matching_values) == 1:
        print('Final Velocity Guess: ',velocity_guess, '\n','Number of Matching Values: ', len(matching_values))

    return matching_values, velocity_guess


def find_good_frame(session_info, skeleton_data, initial_velocity_guess, debug = False):

    sessionID = session_info['sessionID']
    skeleton_type_to_use = session_info['skeleton_type']

    # this_freemocap_session_path = freemocap_data_folder_path / sessionID
    # this_freemocap_data_path = this_freemocap_session_path/'DataArrays'

    # if skeleton_type_to_use == 'mediapipe':
    #     #skeleton_data_path = this_freemocap_data_path/'mediapipe_origin_corrected_and_rotated.npy'
    #     skeleton_data_path = this_freemocap_data_path/'mediaPipeSkel_3d_smoothed.npy'
    #     skeleton_data_path = this_freemocap_data_path/'mediapipe_origin_aligned_skeleton_3D.npy'
    #     skeleton_data = np.load(skeleton_data_path)

    # elif skeleton_type_to_use == 'qualisys':
    #     skeleton_data_path = this_freemocap_data_path/'qualisys_skel_3D.mat'
    #     qualysis_mat_file = sio.loadmat(skeleton_data_path)
    #     skeleton_data = qualysis_mat_file['mat_data_reshaped']
    #     qualisys_num_frames = skeleton_data.shape[0]
        
    #     skeleton_data = skeleton_data[0:int(qualisys_num_frames/2),:,:]


    # else:
    #     print('Please enter a valid skeleton type to use')

    skeleton_velocity_data = np.diff(skeleton_data, axis=0)

    matching_values, velocity_guess = find_best_velocity_guess(skeleton_type_to_use,skeleton_velocity_data, initial_velocity_guess, iteration_range=.1)

    good_frame = matching_values[0]

    if debug:

        figure = plt.figure()
        ax = figure.add_subplot(111, projection = '3d')

        ax.scatter(skeleton_data[good_frame,:,0], skeleton_data[good_frame,:,1], skeleton_data[good_frame,:,2], c='r', marker='o')

        set_axes_ranges(ax, skeleton_data[good_frame,:,:], 1000)

        plt.show()

    return good_frame

if __name__ == '__main__':

        
    this_computer_name = socket.gethostname()

    if this_computer_name == 'DESKTOP-F5LCT4Q':
        #freemocap_validation_data_path = Path(r"C:\Users\aaron\Documents\HumonLab\Spring2022\ValidationStudy\FreeMocap_Data")
        #freemocap_validation_data_path = Path(r'D:\freemocap2022\FreeMocap_Data')
        freemocap_data_folder_path = Path(r'D:\ValidationStudy2022\FreeMocap_Data')

    session_info = {'sessionID': 'gopro_sesh_2022-05-24_16_02_53_JSM_T1_NIH', 'skeleton_type': 'mediapipe'} #name of the sessionID folder    

    #sessionID = 'sesh_2022-05-03_13_43_00_JSM_treadmill_day2_t0'
    #sessionID = 'gopro_sesh_2022-05-24_16_02_53_JSM_T1_NIH'
    #sessionID = 'gopro_sesh_2022-05-24_16_02_53_JSM_T1_WalkRun'
    freemocap_data_array_folder_path = freemocap_data_folder_path/session_info['sessionID']/'DataArrays'

    if session_info['skeleton_type'] == 'mediapipe':
        skeleton_data_path = freemocap_data_array_folder_path/'mediaPipeSkel_3d_smoothed.npy' 
        skeleton_data = np.load(skeleton_data_path)
    elif session_info['skeleton_type'] == 'qualisys':
            skeleton_data_path = freemocap_data_array_folder_path/'qualisysSkel_3d.npy' 
            skeleton_data = np.load(skeleton_data_path)
    good_frame = find_good_frame(session_info, skeleton_data ,initial_velocity_guess=.3, debug = True)
    
    f = 2