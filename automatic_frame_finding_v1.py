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

this_computer_name = socket.gethostname()

if this_computer_name == 'DESKTOP-V3D343U':
    freemocap_validation_data_path = Path(r"I:\My Drive\HuMoN_Research_Lab\FreeMoCap_Stuff\FreeMoCap_Balance_Validation\data")
elif this_computer_name == 'DESKTOP-F5LCT4Q':
    #freemocap_validation_data_path = Path(r"C:\Users\aaron\Documents\HumonLab\Spring2022\ValidationStudy\FreeMocap_Data")
    freemocap_validation_data_path = Path(r'D:\freemocap2022\FreeMocap_Data')
else:
    #freemocap_validation_data_path = Path(r"C:\Users\kiley\Documents\HumonLab\SampleFMC_Data\FreeMocap_Data-20220216T173514Z-001\FreeMocap_Data")
    freemocap_validation_data_path = Path(r"C:\Users\Rontc\Documents\HumonLab\ValidationStudy")

    
sessionID = 'sesh_2022-05-09_12_20_23' #name of the sessionID folder

skeleton_to_plot = 'mediapipe' #for a future situation where we want to rotate openpose/dlc skeletons 
base_frame = 349
debug = False


this_freemocap_session_path = freemocap_validation_data_path / sessionID
this_freemocap_data_path = this_freemocap_session_path/'DataArrays'
save_file = this_freemocap_data_path/'{}_origin_aligned_skeleton_3D.npy'.format(skeleton_to_plot)


if skeleton_to_plot == 'mediapipe':
    #skeleton_data_path = this_freemocap_data_path/'mediapipe_origin_corrected_and_rotated.npy'
    skeleton_data_path = this_freemocap_data_path/'mediaPipeSkel_3d_smoothed.npy'
    right_heel_index = 30
    right_toe_index = 32
    left_heel_index = 29
    left_toe_index = 31

    num_pose_joints = 33

elif skeleton_to_plot == 'openpose':
    skeleton_data_path = this_freemocap_data_path/'openPoseSkel_3d_smoothed.npy'

# primary_foot_indices = [left_heel_index,left_toe_index]
# secondary_foot_index = [right_heel_index]

primary_foot_indices = [left_heel_index,left_toe_index]
secondary_foot_index = [right_heel_index, right_toe_index]

skeleton_data = np.load(skeleton_data_path)

skeleton_velocity_data = np.diff(skeleton_data, axis=0)



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



def find_best_velocity_limit(skeleton_velocity_data, velocity_guess, iteration_range):

    stop_function = False

    skeleton_data_velocity_x_right_heel = skeleton_velocity_data[:,right_heel_index,0]
    skeleton_data_velocity_x_right_toe = skeleton_velocity_data[:,right_toe_index,0]
    skeleton_data_velocity_x_left_heel = skeleton_velocity_data[:,left_heel_index,0]
    skeleton_data_velocity_x_left_toe = skeleton_velocity_data[:,left_toe_index,0]


    right_heel_x_velocity_limits = find_velocity_values_within_limit(skeleton_data_velocity_x_right_heel, velocity_guess)
    right_toe_x_velocity_limits = find_velocity_values_within_limit(skeleton_data_velocity_x_right_toe, velocity_guess)
    left_heel_x_velocity_limits = find_velocity_values_within_limit(skeleton_data_velocity_x_left_heel, velocity_guess)
    left_toe_x_velocity_limits = find_velocity_values_within_limit(skeleton_data_velocity_x_left_toe, velocity_guess)

    matching_values = find_matching_indices_in_lists(right_heel_x_velocity_limits, right_toe_x_velocity_limits, left_heel_x_velocity_limits, left_toe_x_velocity_limits)


    if len(matching_values) > 1 and velocity_guess > 0 and stop_function == False:

        previous__velocity_guess = velocity_guess
        velocity_guess = velocity_guess - iteration_range
        print('Current Velocity Guess: ',velocity_guess, '\n','Number of Matching Values: ', len(matching_values))
        matching_values, velocity_guess = find_best_velocity_limit(skeleton_velocity_data, velocity_guess, iteration_range)

        f = 2
    elif len(matching_values) == 0:
        # stop_function = True
        # velocity_guess = velocity_guess + iteration_range
        # print('Current Velocity Guess: ',velocity_guess, '\n', 'Number of Matching Values: ', len(matching_values))

        # f =2 
        iteration_range = iteration_range/2
        matching_values, velocity_guess = find_best_velocity_limit(skeleton_velocity_data, velocity_guess + 1, iteration_range)

    else:
        print('Final Velocity Guess: ',velocity_guess, '\n','Number of Matching Values: ', len(matching_values))

    return matching_values, velocity_guess


matching_values, final_velocity_guess = find_best_velocity_limit(skeleton_velocity_data, 3.5, iteration_range=1)
f = 2
