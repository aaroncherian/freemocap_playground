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


from fmc_validation_toolbox import good_frame_finder, skeleton_origin_alignment, time_syncing, freemocap_COM_runme, mediapipe_COM_plotting


this_computer_name = socket.gethostname()

if this_computer_name == 'DESKTOP-V3D343U':
    freemocap_validation_data_path = Path(r"I:\My Drive\HuMoN_Research_Lab\FreeMoCap_Stuff\FreeMoCap_Balance_Validation\data")

elif this_computer_name == 'DESKTOP-F5LCT4Q':
    #freemocap_validation_data_path = Path(r"C:\Users\aaron\Documents\HumonLab\Spring2022\ValidationStudy\FreeMocap_Data")
    #freemocap_validation_data_path = Path(r'D:\freemocap2022\FreeMocap_Data')
    freemocap_data_folder_path = Path(r'D:\ValidationStudy2022\FreeMocap_Data')

else:
    #freemocap_validation_data_path = Path(r"C:\Users\kiley\Documents\HumonLab\SampleFMC_Data\FreeMocap_Data-20220216T173514Z-001\FreeMocap_Data")
    freemocap_validation_data_path = Path(r"C:\Users\Rontc\Documents\HumonLab\ValidationStudy")

    
session_one_info = {'sessionID': 'session_SER_1_20_22', 'skeleton_type': 'mediapipe'} #name of the sessionID folder

session_two_info = {'sessionID': 'session_SER_1_20_22', 'skeleton_type': 'qualisys'} 

task = 'time_sync_two_sessions'

session_task_list = [session_one_info]

for session_info in session_task_list:

    if task == 'align_skeleton_with_origin':
        good_frame = good_frame_finder.find_good_frame(session_info,freemocap_data_folder_path, initial_velocity_guess=.2, debug = True)
        skeleton_origin_alignment.align_skeleton_with_origin(session_info, freemocap_data_folder_path, good_frame, debug = True)

    elif task == 'calculate_COM': 
        freemocap_COM_runme.run(session_info, freemocap_data_folder_path)

    elif task == 'time_sync_two_sessions':
        lag = time_syncing.get_time_sync_lag(session_info, session_two_info, freemocap_data_folder_path)
        print(lag)

    elif task == 'align_and_calculate_COM':
        good_frame = good_frame_finder.find_good_frame(session_info,freemocap_data_folder_path, initial_velocity_guess=.2, debug = True)
        skeleton_origin_alignment.align_skeleton_with_origin(session_info, freemocap_data_folder_path, good_frame, debug = True)
        freemocap_COM_runme.run(session_info, freemocap_data_folder_path)

    elif task == 'plot_mediapipe_COM_data':

        stance = 'natural'
        
        if stance == 'natural':
            #num_frame_range = range(9500,12000)
            num_frame_range = range(500,1550) #for BOS
            num_frame_range = range(0, 1550) #for go pro natural 
            #num_frame_range = range(4500,6800)

        elif stance == 'left_leg':
            num_frame_range = range(13000,15180)
            num_frame_range = range(5500,6670)

        elif stance == 'right_leg':
            num_frame_range = range(16000,17450)
            num_frame_range = range(5500,6650)

        camera_fps = 30
        output_video_fps = 30
        tail_length = 120 #number of frames to keep the COM trajectory tail 
        num_frame_range = 0

        COM_plot = mediapipe_COM_plotting.skeleton_COM_Plot(freemocap_data_folder_path,session_info['sessionID'],num_frame_range, camera_fps, output_video_fps, tail_length,stance,static_plot=False)

        this_range_mp_pose_XYZ,this_range_mp_skeleton_segment_XYZ,this_range_segmentCOM_fr_joint_XYZ,this_range_totalCOM_frame_XYZ, video_frames_to_plot = COM_plot.set_up_data()
        COM_plot.generate_plot(this_range_mp_pose_XYZ,this_range_mp_skeleton_segment_XYZ,this_range_segmentCOM_fr_joint_XYZ,this_range_totalCOM_frame_XYZ, video_frames_to_plot)



f = 2