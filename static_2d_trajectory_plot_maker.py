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
    freemocap_validation_data_path = Path(r"C:\Users\aaron\Documents\HumonLab\Spring2022\ValidationStudy\FreeMocap_Data")
    #freemocap_validation_data_path = Path(r'D:\freemocap2022\FreeMocap_Data')
else:
    #freemocap_validation_data_path = Path(r"C:\Users\kiley\Documents\HumonLab\SampleFMC_Data\FreeMocap_Data-20220216T173514Z-001\FreeMocap_Data")
    freemocap_validation_data_path = Path(r"C:\Users\Rontc\Documents\HumonLab\ValidationStudy")
sessionID = 'session_SER_1_20_22' #name of the sessionID folder

save_path = Path(r'C:\Users\aaron\Documents\HumonLab\RMASBM\comparison_graphics')

name = 'fmc_natural_sway.png'
save_image = save_path/name


run_type = 'freemocap'


num_frame_range = range(9400,12200)
# camera_fps = 60
output_video_fps = 60
tail_length = 120 #number of frames to keep the COM trajectory tail 


ax_range = 250

if run_type == 'freemocap':
    from plot_with_classes import skeleton_COM_Plot
    camera_fps = 60
    COM_plot = skeleton_COM_Plot(freemocap_validation_data_path,sessionID,num_frame_range, camera_fps, output_video_fps, tail_length, static_plot=True)
    title_text = 'FreeMoCap/MediaPipe: Natural Stance at Rest'

    left_heel_index = 29
    left_toe_index = 31

    right_heel_index = 30
    right_toe_index = 32
else:
    from qualisys_class_plotting import skeleton_COM_Plot
    camera_fps = 300
    COM_plot = skeleton_COM_Plot(freemocap_validation_data_path,sessionID,num_frame_range, camera_fps, output_video_fps, tail_length)
    title_text = 'Qualisys: Natural Stance at Rest'
    left_heel_index = 18
    left_toe_index = 19

    right_heel_index = 22
    right_toe_index = 23

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
num_frames_to_plot = range(num_frame_range[-1] - num_frame_range[0])

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


figure.suptitle(title_text, fontsize = 16, y = .94)


ax = figure.add_subplot(111)



mx_com = -1*np.nanmean(this_range_totalCOM_frame_XYZ[:,0])
my_com = -1*np.nanmean(this_range_totalCOM_frame_XYZ[:,1])

#for every stance but right its been 300

ax.set_ylim([my_com-ax_range, my_com + ax_range])
ax.set_xlim([mx_com-ax_range,mx_com + ax_range])

#Natural stance

left_heel_mean_x = -1*np.nanmean(left_foot_x[0][:])
left_heel_mean_y = -1*np.nanmean(left_foot_y[0][:])

left_toe_mean_x = -1*np.nanmean(left_foot_x[1][:])
left_toe_mean_y = -1*np.nanmean(left_foot_y[1][:])

right_heel_mean_x = -1*np.nanmean(right_foot_x[0][:])
right_heel_mean_y = -1*np.nanmean(right_foot_y[0][:])

right_toe_mean_x = -1*np.nanmean(right_foot_x[1][:])
right_toe_mean_y = -1*np.nanmean(right_foot_y[1][:])

ax.plot(-1*this_range_totalCOM_frame_XYZ[:,0],-1*this_range_totalCOM_frame_XYZ[:,1], color = 'grey', alpha = .5, label = 'COM Trajectory')
ax.plot([left_heel_mean_x,left_toe_mean_x],[left_heel_mean_y,left_toe_mean_y], color = 'blue', label = 'Left Foot/Lateral Max Bound')
ax.plot([right_heel_mean_x,right_toe_mean_x],[right_heel_mean_y,right_toe_mean_y], color = 'red', label = 'Right Foot/Lateral Min Bound')
ax.plot(np.nanmean(left_foot_avg_x),np.nanmean(left_foot_avg_y), color = 'blue', marker = 'o')
ax.plot(np.nanmean(right_foot_avg_x),np.nanmean(right_foot_avg_y), color = 'red', marker = 'o')

ax.plot([left_heel_mean_x,right_heel_mean_x],[left_heel_mean_y,right_heel_mean_y], color = 'coral', label = 'Posterior Bound', linestyle = '--',alpha = .5)
ax.plot([left_toe_mean_x,right_toe_mean_x],[left_toe_mean_y,right_toe_mean_y], color = 'forestgreen', label = 'Anterior Bound',linestyle = '--',alpha = .5)
ax.plot(np.nanmean(back_foot_avg_x),np.nanmean(back_foot_avg_y), color = 'coral', marker = 'o')
ax.plot(np.nanmean(front_foot_avg_x),np.nanmean(front_foot_avg_y), color = 'forestgreen', marker = 'o')

ax.set_xlabel('X Position (mm)')
ax.set_ylabel('Y Position (mm)')
ax.legend()

f=2 

figure.savefig(save_image)
plt.show()






