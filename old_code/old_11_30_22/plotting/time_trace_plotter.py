

from matplotlib.markers import MarkerStyle
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

from io import BytesIO

from fmc_validation_toolbox.mediapipe_skeleton_builder import mediapipe_indices
from fmc_validation_toolbox.qualisys_skeleton_builder import qualisys_indices

from freemocap_and_qual_plotting import skeleton_data_holder, skeleton_data_for_plotting


this_computer_name = socket.gethostname()
print(this_computer_name)

if this_computer_name == 'DESKTOP-F5LCT4Q':
        #freemocap_validation_data_path = Path(r"C:\Users\aaron\Documents\HumonLab\Spring2022\ValidationStudy\FreeMocap_Data")
        #freemocap_validation_data_path = Path(r'D:\freemocap2022\FreeMocap_Data')
        path_to_freemocap_data_folder = Path(r'D:\ValidationStudy2022\FreeMocap_Data')

#sessionID = 'sesh_2022-05-03_13_43_00_JSM_treadmill_day2_t0' #name of the sessionID folder
#sessionID = 'sesh_2022-05-24_15_55_40_JSM_T1_BOS'
#sessionID = 'gopro_sesh_2022-05-24_16_02_53_JSM_T1_BOS'

session_one_info = {'sessionID':'sesh_2022-05-24_15_55_40_JSM_T1_BOS','skeleton_type':'mediapipe'}
session_two_info = {'sessionID':'qualisys_sesh_2022-05-24_16_02_53_JSM_T1_BOS','skeleton_type':'qualisys'}

#session_one_info = {'sessionID':'sesh_2022-05-24_16_10_46_JSM_T1_WalkRun','skeleton_type':'mediapipe'}
#session_two_info = {'sessionID':'qualisys_sesh_2022-05-24_16_02_53_JSM_T1_WalkRun','skeleton_type':'qualisys'}

stance = 'right_leg'

freemocap_session_path = path_to_freemocap_data_folder / session_one_info['sessionID']

if stance == 'natural':
    #num_frame_range = range(9500,12000)
    #num_frame_range = range(500,1600) #for BOS
    #num_frame_range = range(450, 1300) #for go pro natural 
    #num_frame_range = range(503,1353) #for webcam natural
    
    mediapipe_num_frame_range = (473,1373)

    
    
    #mediapipe_num_frame_range = range(1161,6561)
    qualisys_num_frame_range = range(4730-750,13730-750)
    #qualisys_num_frame_range = range(0,54000,10)
    #num_frame_range = range(4500,6800)

elif stance == 'left_leg':
    num_frame_range = range(13000,15180)
    num_frame_range = range(5500,6670)

elif stance == 'right_leg':
    #num_frame_range = range(16000,17450)
    #num_frame_range = range(5400,6620) #gopro
    num_frame_range = range(5453,6673)
    mediapipe_num_frame_range = range(5423,6653) #for webcam natural
    qualisys_num_frame_range = range(54230-740,66530-740,10)

step_interval = 1

mediapipe_data_holder = skeleton_data_holder(path_to_freemocap_data_folder, session_one_info)
mediapipe_data_to_plot = skeleton_data_for_plotting(mediapipe_data_holder,mediapipe_num_frame_range,1,mediapipe_indices)

qualisys_data_holder = skeleton_data_holder(path_to_freemocap_data_folder, session_two_info)
qualisys_data_to_plot = skeleton_data_for_plotting(qualisys_data_holder,qualisys_num_frame_range,10,qualisys_indices)


fig = plt.figure(figsize=(20,10))

ax_med = fig.add_subplot(2,2,1)
ax_qual = fig.add_subplot(2,2,2)

ax_med.set_title('Mediapipe COM Trajectory')
ax_qual.set_title('Qualisys COM Trajectory')

ax_med_ap = fig.add_subplot(2,2,3)
ax_qual_ap = fig.add_subplot(2,2,4)

num_frames = range(len(mediapipe_data_to_plot.total_body_COM_data))
time_array = [frame/30 for frame in num_frames]

ax_med.set_ylim([-175,175])
ax_med.set_xlim([time_array[0],time_array[-1]])
ax_qual.set_ylim([-175,175])
ax_qual.set_xlim([time_array[0],time_array[-1]])

ax_med.axes.xaxis.set_ticks([])
ax_qual.axes.xaxis.set_ticks([])

#ax_med.set_xlabel('Time (s)')
ax_med.set_ylabel('X Position (mm)')
#ax_qual.set_xlabel('Time (s)')
ax_qual.set_ylabel('X Position (mm)')

ax_med_ap.set_xlim([time_array[0],time_array[-1]])
ax_med_ap.set_ylim([-100,300])

ax_qual_ap.set_xlim([time_array[0],time_array[-1]])
ax_qual_ap.set_ylim([-100,300])

ax_med_ap.set_xlabel('Time (s)')
ax_med_ap.set_ylabel('Y Position (mm)')

ax_qual_ap.set_xlabel('Time (s)')
ax_qual_ap.set_ylabel('Y Position (mm)')

# ##Natural
# ax_med.plot(time_array,mediapipe_data_to_plot.left_foot_average_XYZ[:,0], color = 'blue')
# ax_med.plot(time_array,mediapipe_data_to_plot.right_foot_average_XYZ[:,0], color = 'red')
# ax_med.plot(time_array,mediapipe_data_to_plot.total_body_COM_data[:,0],color='grey')

# ax_qual.plot(time_array,qualisys_data_to_plot.left_foot_average_XYZ[:,0], color = 'blue')
# ax_qual.plot(time_array,qualisys_data_to_plot.right_foot_average_XYZ[:,0], color = 'red')
# ax_qual.plot(time_array,qualisys_data_to_plot.total_body_COM_data[:,0],color='grey')

# ax_med_ap.plot(time_array,mediapipe_data_to_plot.front_foot_avg_position[:,1], color = 'forestgreen')
# ax_med_ap.plot(time_array,mediapipe_data_to_plot.back_foot_avg_position[:,1], color = 'coral')
# ax_med_ap.plot(time_array,mediapipe_data_to_plot.total_body_COM_data[:,1],color='grey')

# ax_qual_ap.plot(time_array,qualisys_data_to_plot.front_foot_avg_position[:,1], color = 'forestgreen')
# ax_qual_ap.plot(time_array,qualisys_data_to_plot.back_foot_avg_position[:,1], color = 'coral')
# ax_qual_ap.plot(time_array,qualisys_data_to_plot.total_body_COM_data[:,1],color='grey')

##Natural
#ax_med.plot(time_array,mediapipe_data_to_plot.left_foot_average_XYZ[:,0], color = 'blue')
ax_med.plot(time_array,mediapipe_data_to_plot.right_foot_average_XYZ[:,0], color = 'red')
ax_med.plot(time_array,mediapipe_data_to_plot.total_body_COM_data[:,0],color='grey')

#ax_qual.plot(time_array,qualisys_data_to_plot.left_foot_average_XYZ[:,0], color = 'blue')
ax_qual.plot(time_array,qualisys_data_to_plot.right_foot_average_XYZ[:,0], color = 'red')
ax_qual.plot(time_array,qualisys_data_to_plot.total_body_COM_data[:,0],color='grey')

ax_med_ap.plot(time_array,mediapipe_data_to_plot.front_foot_avg_position[:,1], color = 'forestgreen')
ax_med_ap.plot(time_array,mediapipe_data_to_plot.back_foot_avg_position[:,1], color = 'coral')
ax_med_ap.plot(time_array,mediapipe_data_to_plot.total_body_COM_data[:,1],color='grey')

ax_qual_ap.plot(time_array,qualisys_data_to_plot.front_foot_avg_position[:,1], color = 'forestgreen')
ax_qual_ap.plot(time_array,qualisys_data_to_plot.back_foot_avg_position[:,1], color = 'coral')
ax_qual_ap.plot(time_array,qualisys_data_to_plot.total_body_COM_data[:,1],color='grey')
plt.show()


save_path = r'C:\Users\aaron\Documents\HumonLab\DynamicWalking2022\presentation_graphics\COM_rightleg_white.jpeg'
fig.savefig(save_path, transparent = 'False')
