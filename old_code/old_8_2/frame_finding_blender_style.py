

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

    
sessionID = 'sesh_2022-05-12_13_43_40' #name of the sessionID folder
#sessionID = 'sesh_2022-05-09_15_40_59'

skeleton_to_plot = 'mediapipe' #for a future situation where we want to rotate openpose/dlc skeletons 


this_freemocap_session_path = freemocap_validation_data_path / sessionID
this_freemocap_data_path = this_freemocap_session_path/'DataArrays'
save_file = this_freemocap_data_path/'{}_origin_aligned_skeleton_3D.npy'.format(skeleton_to_plot)


if skeleton_to_plot == 'mediapipe':
    #skeleton_data_path = this_freemocap_data_path/'mediapipe_origin_corrected_and_rotated.npy'
    skeleton_data_path = this_freemocap_data_path/'mediaPipeSkel_3d_smoothed_unrotated.npy'
    right_heel_index = 30
    right_toe_index = 32
    left_heel_index = 29
    left_toe_index = 31


skeleton_data = np.load(skeleton_data_path)


frame_nan_counts = []
frame_mean_reproj_error = []

for this_frame in range(skeleton_data.shape[0]):
    frame_nan_counts.append(np.sum(np.isnan(skeleton_data[this_frame,:,0])))
    frame_mean_reproj_error.append(np.nanmean(skeleton_data[this_frame,:]))

nan_times_vis = np.array(frame_nan_counts)*np.array(frame_mean_reproj_error)
num_frames  = len(frame_nan_counts)
# nan_times_vis[0:int(num_frames/5)] = np.nanmax(nan_times_vis)
# nan_times_vis[-int(num_frames/5):-1] = np.nanmax(nan_times_vis)

good_clean_frame_number = np.nanargmin(nan_times_vis) # the frame with the fewest nans (i.e. hopefully a frame where all tracked points are visible)

f = 2