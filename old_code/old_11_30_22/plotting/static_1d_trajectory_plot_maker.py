# from plot_with_classes import skeleton_COM_Plot
# from qualisys_class_plotting import skeleton_COM_Plot



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


this_computer_name = socket.gethostname()
print(this_computer_name)


if this_computer_name == 'DESKTOP-V3D343U':
    freemocap_validation_data_path = Path(r"I:\My Drive\HuMoN_Research_Lab\FreeMoCap_Stuff\FreeMoCap_Balance_Validation\data")
elif this_computer_name == 'DESKTOP-F5LCT4Q':
    #freemocap_validation_data_path = Path(r"C:\Users\aaron\Documents\HumonLab\Spring2022\ValidationStudy\FreeMocap_Data")
    #freemocap_validation_data_path = Path(r'D:\freemocap2022\FreeMocap_Data')
    #freemocap_validation_data_path = Path(r'D:\ValidationStudy_aaron\FreeMoCap_Data')
    freemocap_validation_data_path = Path(r'D:\ValidationStudy2022\FreeMocap_Data')

else:
    #freemocap_validation_data_path = Path(r"C:\Users\kiley\Documents\HumonLab\SampleFMC_Data\FreeMocap_Data-20220216T173514Z-001\FreeMocap_Data")
    freemocap_validation_data_path = Path(r"C:\Users\Rontc\Documents\HumonLab\ValidationStudy")

#sessionID = 'session_SER_1_20_22' #name of the sessionID folder
#sessionID = 'gopro_sesh_2022-05-24_16_02_53_JSM_T1_NIH'
#sessionID = 'sesh_2022-05-24_16_02_53_JSM_T1_NIH'
#sessionID = 'gopro_sesh_2022-05-24_16_02_53_JSM_T1_BOS'
#sessionID = 'qualisys_sesh_2022-05-24_16_02_53_JSM_T1_BOS'

sessionID = 'sesh_2022-05-24_16_02_53_JSM_T1_NIH'

#sessionID = 'sesh_2022-05-03_13_43_00_JSM_treadmill_day2_t0'
#save_path = Path(r'D:\ValidationStudy_aaron\FreeMoCap_Data\sesh_2022-11-02_13_55_55_atc_nih_balance\data_analysis\analysis_2023-01-27_16_38_20')

save_path = Path(r'D:\ValidationStudy2022\FreeMocap_Data\sesh_2022-05-24_16_02_53_JSM_T1_NIH\data_analysis\analysis_2023-01-30_13_12_51')



#run_type = 'mediapipe'
run_type = 'mediapipe'

# run_type = 'freemocap'
stance = 'natural'

title = 'COM_X_Y_EC_SG'

name = f'{title}.png'
#name = '{}_{}_sway.png'.format(run_type, stance)
save_image = save_path/name

#num_frame_range = range(9900,12000)
# camera_fps = 60
output_video_fps = 30
tail_length = 120 #number of frames to keep the COM trajectory tail 


ax_range = 300



if run_type == 'mediapipe':
    from old_code.old_8_2.plot_with_classes import skeleton_COM_Plot

    if stance == 'natural':
        #num_frame_range = range(9500,12000)
        #num_frame_range = range(0,1550)
        num_frame_range = range(2750,4400)
        ax_range = 300


    if stance == 'left_leg':
        num_frame_range = range(13000,15180)
        ax_range = 550
        left_leg_down_ranges = [[13350,14000],[14370,14550],[15000,15180]]

    if stance == 'right_leg':
        #num_frame_range = range(16680,17740)
        num_frame_range = range(16000,17450)
        right_leg_down_ranges = [[16290,16670],[17000,17310]]
        ax_range = 650



    camera_fps = 60
    COM_plot = skeleton_COM_Plot(freemocap_validation_data_path,sessionID,num_frame_range, camera_fps, output_video_fps, tail_length, stance = 'natural', static_plot=True)
    title_text = title

    left_heel_index = 29
    left_toe_index = 31

    right_heel_index = 30
    right_toe_index = 32



elif run_type == 'qualisys':
    from qualisys_class_plotting import skeleton_COM_Plot

    if stance == 'natural':
        num_frame_range = range(0,60000)
        ax_range = 300
    
    if stance == 'left_leg':
        num_frame_range = range(75850,86750)
        ax_range = 550
        mp_left_leg_down_ranges =[[13350,14000],[14375,14550],[15005,15180]]

        left_leg_down_ranges =[]
        for foot_down_range in mp_left_leg_down_ranges:
            start_point = foot_down_range[0] - 13000
            end_point = foot_down_range[1] - 13000
            left_leg_down_ranges.append([start_point*5,end_point*5])


    if stance == 'right_leg':   
        #num_frame_range = range(90890,99685)
        num_frame_range = range(90890,98140)
        ax_range = 600
        mp_right_leg_down_ranges = [[16290,16670],[17050,17310]]
        right_leg_down_ranges = []
        for foot_down_range in mp_right_leg_down_ranges:
            start_point = foot_down_range[0] - 16000 
            end_point = foot_down_range[1] - 16000
            right_leg_down_ranges.append([start_point*5,end_point*5]) 



    camera_fps = 300
    COM_plot = skeleton_COM_Plot(freemocap_validation_data_path,sessionID,num_frame_range, camera_fps, output_video_fps, tail_length, stance = 'natural')
    title_text = 'Qualisys Right Leg Stance Sway'
    left_heel_index = 18
    left_toe_index = 19

    right_heel_index = 22
    right_toe_index = 23

num_frames_to_plot = range(num_frame_range[-1] - num_frame_range[0])
#num_frame_range = range(frame_range_for_trajectory[0],9950)


#num_frame_range = 0

#frame_to_plot = 11565


# COM_plot = skeleton_COM_Plot(freemocap_validation_data_path,sessionID,num_frame_range, camera_fps, output_video_fps, tail_length, static_plot=True)
#COM_plot = skeleton_COM_Plot(freemocap_validation_data_path,sessionID,num_frame_range, camera_fps, output_video_fps, tail_length)

this_range_mp_pose_XYZ,this_range_mp_skeleton_segment_XYZ,this_range_segmentCOM_fr_joint_XYZ,this_range_totalCOM_frame_XYZ, video_frames_to_plot = COM_plot.set_up_data()

skel_x = this_range_mp_pose_XYZ[:,:,0]
skel_y = this_range_mp_pose_XYZ[:,:,1]
skel_z = this_range_mp_pose_XYZ[:,:,2]

# left_foot_position_XYZ, left_foot_avg_position_XYZ = COM_plot.get_foot_data(skel_x,skel_y,skel_z,29,31)
# right_foot_position_XYZ, right_foot_avg_position_XYZ = COM_plot.get_foot_data(skel_x,skel_y,skel_z,30,32)

left_foot_position_XYZ, left_foot_avg_position_XYZ = COM_plot.get_foot_data(skel_x,skel_y,skel_z,left_heel_index,left_toe_index)
right_foot_position_XYZ, right_foot_avg_position_XYZ = COM_plot.get_foot_data(skel_x,skel_y,skel_z,right_heel_index,right_toe_index)

front_foot_avg_position_XYZ, back_foot_avg_position_XYZ = COM_plot.get_anterior_posterior_bounds(left_foot_position_XYZ,right_foot_position_XYZ)


time_array = []
for frame_num in num_frames_to_plot:
    time_array.append(frame_num/camera_fps)

left_foot_x = left_foot_position_XYZ[0]
left_foot_y = left_foot_position_XYZ[1]
left_foot_z = left_foot_position_XYZ[2]

right_foot_x = right_foot_position_XYZ[0]
right_foot_y = right_foot_position_XYZ[1]
right_foot_z = right_foot_position_XYZ[2]



left_foot_avg_x = -1*left_foot_avg_position_XYZ[0]
left_foot_avg_y = -1*left_foot_avg_position_XYZ[1]
left_foot_avg_z = left_foot_avg_position_XYZ[2]

right_foot_avg_x = -1*right_foot_avg_position_XYZ[0]
right_foot_avg_y = -1*right_foot_avg_position_XYZ[1]
right_foot_avg_z = right_foot_avg_position_XYZ[2]


back_foot_avg_x = -1*back_foot_avg_position_XYZ[0]
back_foot_avg_y = -1*back_foot_avg_position_XYZ[1]
back_foot_avg_z = back_foot_avg_position_XYZ[2]

front_foot_avg_x = -1*front_foot_avg_position_XYZ[0]
front_foot_avg_y = -1*front_foot_avg_position_XYZ[1]
front_foot_avg_z = front_foot_avg_position_XYZ[2]



figure = plt.figure(figsize= (10,10))


#figure.suptitle(title_text, fontsize = 16, y = .94)


ax = figure.add_subplot(211)
ax2 = figure.add_subplot(212)

ax.set_ylabel('X Position (mm)', fontsize = 25)
ax2.set_ylabel('Y Position (mm)', fontsize = 25)
ax2.set_xlabel('Time (s)', fontsize = 25)
ax2.tick_params(axis='x', labelsize= 15)
ax.tick_params(axis='y', labelsize= 15)
ax2.tick_params(axis='y', labelsize= 15)

mx_com = -1*np.nanmean(this_range_totalCOM_frame_XYZ[:,0])
my_com = -1*np.nanmean(this_range_totalCOM_frame_XYZ[:,1])

#for every stance but right its been 300


# if stance == 'natural':
#     ax.set_ylim([-460,140])
#     ax2.set_ylim([-340,260])
# if stance == 'left_leg':
#     ax.set_ylim([-590, 510])
#     ax2.set_ylim([-690, 409])
# if stance == 'right_leg':
#     ax.set_ylim([-899, 401])
#     ax2.set_ylim([-739, 561])

#ax.set_xlim([time_array[0],time_array[-1]])
#ax2.set_xlim([time_array[0],time_array[-1]])
#ax.set_ylim([mx_com-ax_range, mx_com + ax_range])


##Reset for time marking
#ax.set_xlim([time_array[0],time_array[-1]])
#ax2.set_xlim([time_array[0],time_array[-1]])
#ax.axes.xaxis.set_ticks([])
#ax.set_xlim([num_frame_range[0],num_frame_range[-1]])


# 6+
#Natural stance
if stance == 'natural':
    # ax.plot(time_array,-1*this_range_totalCOM_frame_XYZ[:,0], color = 'grey')
    # ax.plot(time_array, left_foot_avg_x, color = 'blue')
    # ax.plot(time_array, right_foot_avg_x, color = 'red')

    # ax2.plot(time_array,-1*this_range_totalCOM_frame_XYZ[:,1], color = 'grey')
    # ax2.plot(time_array,front_foot_avg_y, color = 'forestgreen')
    # ax2.plot(time_array,back_foot_avg_y, color = 'coral')
    
    ax.plot(-1*this_range_totalCOM_frame_XYZ[:,0], color = 'grey')
    ax.plot(left_foot_avg_x, color = 'blue')
    ax.plot(right_foot_avg_x, color = 'red')

    ax2.plot(time_array,-1*this_range_totalCOM_frame_XYZ[:,1], color = 'grey')
    ax2.plot(time_array,front_foot_avg_y, color = 'forestgreen')
    ax2.plot(time_array,back_foot_avg_y, color = 'coral')

elif stance == 'left_leg':
    ax.plot(time_array,-1*this_range_totalCOM_frame_XYZ[:,0], color = 'grey')
    ax.plot(time_array, left_foot_avg_x, color = 'blue')

    ax2.plot(time_array,-1*this_range_totalCOM_frame_XYZ[:,1], color = 'grey')
    ax2.plot(time_array,-1*left_foot_y[1], color = 'forestgreen')
    ax2.plot(time_array,-1*left_foot_y[0], color = 'coral')
    
    for foot_down_range in left_leg_down_ranges:

        if run_type == 'freemocap':
            start_point = foot_down_range[0] -num_frame_range[0]
            end_point = foot_down_range[1] -num_frame_range[0]
        if run_type == 'qualisys':
            start_point = foot_down_range[0]
            end_point = foot_down_range[1]
        ax.plot(time_array[start_point:end_point],right_foot_avg_x[start_point:end_point], color = 'red')

elif stance == 'right_leg':
    ax.plot(time_array,-1*this_range_totalCOM_frame_XYZ[:,0], color = 'grey')
    ax.plot(time_array, right_foot_avg_x, color = 'red')

    ax2.plot(time_array,-1*this_range_totalCOM_frame_XYZ[:,1], color = 'grey')
    ax2.plot(time_array,-1*right_foot_y[1], color = 'forestgreen')
    ax2.plot(time_array,-1*right_foot_y[0], color = 'coral')

        
    for foot_down_range in right_leg_down_ranges:

        if run_type == 'freemocap':           
            start_point = foot_down_range[0] - num_frame_range[0]
            end_point = foot_down_range[1] - num_frame_range[0]
        
        if run_type == 'qualisys':
            start_point = foot_down_range[0] 
            end_point = foot_down_range[1] 
        ax.plot(time_array[start_point:end_point],left_foot_avg_x[start_point:end_point], color = 'blue')
        f=2



figure.savefig(save_image)
plt.show()

f=2





