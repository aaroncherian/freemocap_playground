from plot_with_classes import skeleton_COM_Plot
#from qualisys_class_plotting import skeleton_COM_Plot


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


#num_frame_range = range(frame_range_for_trajectory[0],9950)

num_frame_range = 0
camera_fps = 60
output_video_fps = 60
tail_length = 120 #number of frames to keep the COM trajectory tail 
#num_frame_range = 0

frame_to_plot = 15546


COM_plot = skeleton_COM_Plot(freemocap_validation_data_path,sessionID,num_frame_range, camera_fps, output_video_fps, tail_length, stance = 'natural',static_plot=True)

this_range_mp_pose_XYZ,this_range_mp_skeleton_segment_XYZ,this_range_segmentCOM_fr_joint_XYZ,this_range_totalCOM_frame_XYZ, video_frames_to_plot = COM_plot.set_up_data()

skel_x = this_range_mp_pose_XYZ[:,:,0]
skel_y = this_range_mp_pose_XYZ[:,:,1]
skel_z = this_range_mp_pose_XYZ[:,:,2]

left_foot_position_XYZ, left_foot_avg_position_XYZ = COM_plot.get_foot_data(skel_x,skel_y,skel_z,29,31)
right_foot_position_XYZ, right_foot_avg_position_XYZ = COM_plot.get_foot_data(skel_x,skel_y,skel_z,30,32)

front_foot_avg_position_XYZ, back_foot_avg_position_XYZ = COM_plot.get_anterior_posterior_bounds(left_foot_position_XYZ,right_foot_position_XYZ)


this_frame_segment_COM_x = -1*this_range_segmentCOM_fr_joint_XYZ[frame_to_plot,:,0]
this_frame_segment_COM_y = -1*this_range_segmentCOM_fr_joint_XYZ[frame_to_plot,:,1]
this_frame_segment_COM_z = this_range_segmentCOM_fr_joint_XYZ[frame_to_plot,:,2]

this_frame_total_COM_x = -1*this_range_totalCOM_frame_XYZ[frame_to_plot,0]
this_frame_total_COM_y = -1*this_range_totalCOM_frame_XYZ[frame_to_plot,1]
this_frame_total_COM_z = this_range_totalCOM_frame_XYZ[frame_to_plot,2]

this_frame_bones_XYZ = this_range_mp_skeleton_segment_XYZ[frame_to_plot]

this_frame_skel_x = -1*skel_x[frame_to_plot]
this_frame_skel_y = -1*skel_y[frame_to_plot]
this_frame_skel_z = skel_z[frame_to_plot]

left_foot_x = left_foot_position_XYZ[0]
left_foot_y = left_foot_position_XYZ[1]
left_foot_z = left_foot_position_XYZ[2]

right_foot_x = right_foot_position_XYZ[0]
right_foot_y = right_foot_position_XYZ[1]
right_foot_z = right_foot_position_XYZ[2]

this_frame_left_foot_x = [-1*left_foot_x[joint][frame_to_plot] for joint in range(2)]
this_frame_left_foot_y = [-1*left_foot_y[joint][frame_to_plot] for joint in range(2)]
this_frame_left_foot_z = [left_foot_z[joint][frame_to_plot] for joint in range(2)]

this_frame_right_foot_x = [-1*right_foot_x[joint][frame_to_plot] for joint in range(2)]
this_frame_right_foot_y = [-1*right_foot_y[joint][frame_to_plot] for joint in range(2)]
this_frame_right_foot_z = [right_foot_z[joint][frame_to_plot] for joint in range(2)]



left_foot_avg_x = -1*left_foot_avg_position_XYZ[0]
left_foot_avg_y = -1*left_foot_avg_position_XYZ[1]
left_foot_avg_z = left_foot_avg_position_XYZ[2]

right_foot_avg_x = -1*right_foot_avg_position_XYZ[0]
right_foot_avg_y = -1*right_foot_avg_position_XYZ[1]
right_foot_avg_z = right_foot_avg_position_XYZ[2]

this_frame_left_foot_avg_x = left_foot_avg_x[frame_to_plot]
this_frame_left_foot_avg_y = left_foot_avg_y[frame_to_plot]
this_frame_left_foot_avg_z = left_foot_avg_z[frame_to_plot]

this_frame_right_foot_avg_x = right_foot_avg_x[frame_to_plot]
this_frame_right_foot_avg_y = right_foot_avg_y[frame_to_plot]
this_frame_right_foot_avg_z = right_foot_avg_z[frame_to_plot]

back_foot_avg_x = -1*back_foot_avg_position_XYZ[0]
back_foot_avg_y = -1*back_foot_avg_position_XYZ[1]
back_foot_avg_z = back_foot_avg_position_XYZ[2]

front_foot_avg_x = -1*front_foot_avg_position_XYZ[0]
front_foot_avg_y = -1*front_foot_avg_position_XYZ[1]
front_foot_avg_z = front_foot_avg_position_XYZ[2]

# this_frame_back_foot_avg_x = back_foot_avg_x[frame_to_plot]
# this_frame_back_foot_avg_y = back_foot_avg_y[frame_to_plot]
# this_frame_back_foot_avg_z = back_foot_avg_z[frame_to_plot]

# this_frame_front_foot_avg_x = front_foot_avg_x[frame_to_plot]
# this_frame_front_foot_avg_y = front_foot_avg_y[frame_to_plot]
# this_frame_front_foot_avg_z = front_foot_avg_z[frame_to_plot]






left_shoulder_joint = this_frame_bones_XYZ['left_upper_arm'][0]
right_shoulder_joint = this_frame_bones_XYZ['right_upper_arm'][0]
shoulder_connection_x, shoulder_connection_y, shoulder_connection_z = [left_shoulder_joint[0],right_shoulder_joint[0]],[left_shoulder_joint[1],right_shoulder_joint[1]],[left_shoulder_joint[2],right_shoulder_joint[2]]

#plotting a line across the hips

left_hip_joint = this_frame_bones_XYZ['left_thigh'][0]
right_hip_joint = this_frame_bones_XYZ['right_thigh'][0]
hip_connection_x, hip_connection_y, hip_connection_z = [left_hip_joint[0],right_hip_joint[0]],[left_hip_joint[1],right_hip_joint[1]],[left_hip_joint[2],right_hip_joint[2]]



figure = plt.figure(figsize= (10,10))
figure.suptitle('FreeMoCap Center of Mass', fontsize = 16, y = .94, color = 'royalblue')

ax = figure.add_subplot(111, projection = '3d')
ax.view_init(elev = 20, azim = -121)

ax_range = 800
mx = -1*np.nanmean(skel_x[int(frame_to_plot),:])
my = -1*np.nanmean(skel_y[int(frame_to_plot),:])
mz = np.nanmean(skel_z[int(frame_to_plot),:])


ax.set_xlim([mx-ax_range, mx+ax_range]) #maybe set ax limits before the function? if we're using cla() they probably don't need to be redefined every time 
ax.set_ylim([my-ax_range, my+ax_range])
ax.set_zlim([mz-ax_range, mz+ax_range])

for segment in this_frame_bones_XYZ.keys():
    prox_joint = this_frame_bones_XYZ[segment][0] 
    dist_joint = this_frame_bones_XYZ[segment][1]
    
    bone_x,bone_y,bone_z = [prox_joint[0],dist_joint[0]],[prox_joint[1],dist_joint[1]],[prox_joint[2],dist_joint[2]] 

    bone_x = [-x for x in bone_x]
    bone_y = [-x for x in bone_y]

    ax.plot(bone_x,bone_y,bone_z,color = 'black')

# for segment in this_frame_bones_XYZ.keys():
#     if segment == 'head':
#         head_point = this_frame_bones_XYZ[segment][0]
#         bone_x,bone_y,bone_z = [head_point[0],head_point[1],head_point[2]]

#         bone_x = -bone_x
#         bone_y = -bone_y
#     else:
#         prox_joint = this_frame_bones_XYZ[segment][0] 
#         dist_joint = this_frame_bones_XYZ[segment][1]

#         dist_joint = this_frame_bones_XYZ[segment][1]              
#         bone_x,bone_y,bone_z = [prox_joint[0],dist_joint[0]],[prox_joint[1],dist_joint[1]],[prox_joint[2],dist_joint[2]] 
        
#         bone_x = [-x for x in bone_x]
#         bone_y = [-x for x in bone_y]
#     ax.plot(bone_x,bone_y,bone_z,color = 'black')

ax.plot(-1*np.array(shoulder_connection_x),-1*np.array(shoulder_connection_y),shoulder_connection_z,color = 'black')
ax.plot(-1*np.array(hip_connection_x), -1*np.array(hip_connection_y), hip_connection_z, color = 'black')


ax.scatter(this_frame_segment_COM_x,this_frame_segment_COM_y,this_frame_segment_COM_z, color = 'orange', label = 'Segment Center of Mass')    
ax.scatter(this_frame_total_COM_x,this_frame_total_COM_y,this_frame_total_COM_z, color = 'magenta', label = 'Total Body Center of Mass', marker = '*', s = 70, edgecolor = 'purple')
ax.scatter(this_frame_skel_x, this_frame_skel_y,this_frame_skel_z, color = 'grey', alpha = .3)

#ax.plot(left_foot_x,left_foot_y,left_foot_z, color = 'blue')
#ax.plot(right_foot_x,right_foot_y,right_foot_z, color = 'red')
ax.plot(this_frame_left_foot_x,this_frame_left_foot_y,this_frame_left_foot_z, color = 'blue')
ax.plot(this_frame_right_foot_x,this_frame_right_foot_y,this_frame_right_foot_z, color = 'red')

# ax.scatter(this_frame_total_COM_x,this_frame_total_COM_y,0, color = 'magenta', marker = '*', s = 70, edgecolor = 'purple', alpha = .5)
# ax.plot([this_frame_total_COM_x,this_frame_total_COM_x],[this_frame_total_COM_y,this_frame_total_COM_y],[0,this_frame_total_COM_z], color = 'grey', linestyle = '--',alpha = .5)
# # ax.legend(bbox_to_anchor=(.5, 0.14))

# ax.plot([this_frame_left_foot_x[0],this_frame_right_foot_x[0]],[this_frame_left_foot_y[0],this_frame_right_foot_y[0]],[this_frame_left_foot_z[0],this_frame_right_foot_z[0]], color = 'coral', linestyle = '--',alpha = .5)
# ax.plot([this_frame_left_foot_x[1],this_frame_right_foot_x[1]],[this_frame_left_foot_y[1],this_frame_right_foot_y[1]],[this_frame_left_foot_z[1],this_frame_right_foot_z[1]], color = 'forestgreen', linestyle = '--',alpha = .5)

# frame_range_for_trajectory = range(9900,12300)
# xx = np.linspace(my-ax_range,my+ax_range,len(frame_range_for_trajectory))

# current_frame = frame_to_plot - frame_range_for_trajectory[0]

# distance_raised = 1600
# ax.plot(xx,-1*this_range_totalCOM_frame_XYZ[frame_range_for_trajectory[0]:frame_range_for_trajectory[-1]+1,1]+distance_raised,mx-ax_range,zdir ='x', color = 'grey',alpha = .5)
# ax.plot([xx[current_frame],xx[current_frame]],[front_foot_avg_y[frame_range_for_trajectory[-1]]+distance_raised,back_foot_avg_y[frame_range_for_trajectory[-1]]+distance_raised], mx-ax_range,zdir ='x', color = 'black',alpha = .7)
# ax.plot(xx, front_foot_avg_y[frame_range_for_trajectory[0]:frame_range_for_trajectory[-1]+1]+distance_raised, mx-ax_range,zdir ='x', color = 'forestgreen',alpha = .5)
# ax.plot(xx, back_foot_avg_y[frame_range_for_trajectory[0]:frame_range_for_trajectory[-1]+1]+distance_raised, mx-ax_range,zdir ='x', color = 'coral',alpha = .5)

# ax.plot(xx[0:current_frame],-1*this_range_totalCOM_frame_XYZ[frame_range_for_trajectory[0]:frame_to_plot,1]+distance_raised,mx-ax_range,zdir ='x', color = 'black',alpha = .8)
# ax.scatter(xx[current_frame],-1*this_range_totalCOM_frame_XYZ[frame_to_plot,1]+distance_raised,mx-ax_range,zdir ='x', color = 'magenta',alpha = .8,marker = '*', edgecolor = 'purple', s = 40)

#figure.savefig('path/to/save/image/to.png')
plt.show()

f = 2

#COM_plot.generate_plot(this_range_mp_pose_XYZ,this_range_mp_skeleton_segment_XYZ,this_range_segmentCOM_fr_joint_XYZ,this_range_totalCOM_frame_XYZ, video_frames_to_plot)
