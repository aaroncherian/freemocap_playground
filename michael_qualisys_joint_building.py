import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path 
import socket
import pickle
import scipy.io as sio

import pandas as pd

qualisys_joints = [
'left_wrist',
'cspine',
'left_shoulder',
'right_shoulder',
]
qualisys_marker_labels = [
'HeadTop',	
'HeadFront',
'HeadLeft',
'HeadRight',
'R_AntShoulder',
'R_PostShoulder',
'R_Arm',
'R_LatElbow',
'R_MedElbow',
'R_LatHand',
'R_MedHand',
'R_Hand',
'R_Thigh',
'R_LatKnee',
'R_MedKnee',
'R_Shin',
'R_LatAnkle',
'R_MedAnkle',
'R_Heel',
'R_LatFoot',
'R_MedFoot',
'R_Toe',
'L_AntShoulder',
'L_PostShoulder',
'L_LatElbow',
'L_MedElbow',
'L_LatHand',
'L_MedHand',
'L_Hand',
'L_Thigh',
'L_LatKnee',
'L_MedKnee',
'L_Shin',
'L_LatAnkle',
'L_MedAnkle',
'L_Heel',
'L_LatFoot',
'L_MedFoot',
'L_Toe',
'R_Back',
'L_Back',
'RPSIS',
'LPSIS',
'RASIS',
'LASIS',
'Chest']


def set_axes_ranges(plot_ax,skeleton_data,ax_range):

    mx = np.nanmean(skeleton_data[:,0])
    my = np.nanmean(skeleton_data[:,1])
    mz = np.nanmean(skeleton_data[:,2])

    plot_ax.set_xlim(mx-ax_range,mx+ax_range)
    plot_ax.set_ylim(my-ax_range,my+ax_range)
    plot_ax.set_zlim(mz-ax_range,mz+ax_range)     


#1) Set your data path using your computer name
#2) Make an empty session folder with a DataArrays folder in it
#3) Put in the original qualisys MAT file 
#4) Put the matlab script in the DataArrays folder 
#5) Run the script and save out the reorganized .mat file
#6) Change the session ID in the code 
#7) Pick a frame to use 


this_computer_name = socket.gethostname()
print(this_computer_name)

if this_computer_name == 'DESKTOP-V3D343U':
    freemocap_data_path = Path(r"I:\My Drive\HuMoN_Research_Lab\FreeMoCap_Stuff\FreeMoCap_Balance_Validation\data")
elif this_computer_name == 'DESKTOP-F5LCT4Q':
    #freemocap_data_path = Path(r"C:\Users\aaron\Documents\HumonLab\Spring2022\ValidationStudy\FreeMocap_Data")
    #freemocap_data_path = Path(r'D:\freemocap2022\FreeMocap_Data')
    freemocap_data_folder_path = Path(r'D:\ValidationStudy2022\FreeMocap_Data')
elif this_computer_name == 'michaels computer':
    freemocap_data_folder_path = Path(r'insert_path_here')
else:
    #freemocap_validation_data_path = Path(r"C:\Users\kiley\Documents\HumonLab\SampleFMC_Data\FreeMocap_Data-20220216T173514Z-001\FreeMocap_Data")
    freemocap_data_path = Path(r"C:\Users\Rontc\Documents\HumonLab\ValidationStudy")

session_ID = 'sesh_jon_treadmill'
debug = True
frame_to_use = 5000
freemocap_data_array_path = freemocap_data_folder_path/session_ID/'DataArrays'

qualisys_data_path = freemocap_data_array_path/'mat_data_reshaped.mat'
qualisys_mat_file = sio.loadmat(qualisys_data_path)

qualisys_data = qualisys_mat_file['mat_data_reshaped']

num_frames = qualisys_data.shape[0]
num_markers = len(qualisys_joints)
qualisys_joints_array = np.empty([num_frames,num_markers,3])


##Left wrist
left_wrist1 = qualisys_marker_labels.index('L_LatHand') #getting the index of the marker
left_wrist2 = qualisys_marker_labels.index('L_MedHand')

left_qualisys_wrist1_XYZ = qualisys_data[frame_to_use,left_wrist1,:] #using the marker index to get the XYZ position of that marker
left_qualisys_wrist2_XYZ = qualisys_data[frame_to_use,left_wrist2,:]

wrist_XYZ = np.mean([left_qualisys_wrist1_XYZ,left_qualisys_wrist2_XYZ],axis=0) #calculating the joint XYZ position

left_wrist_joint_index = qualisys_joints.index('left_wrist') #getting the index of the joint
qualisys_joints_array[frame_to_use,left_wrist_joint_index,:] = wrist_XYZ #adding the XYZ position to the joint array using the joint index 


##Cervical spine
right_antshoulder_index = qualisys_marker_labels.index('R_AntShoulder')
right_postshoulder_index = qualisys_marker_labels.index('R_PostShoulder')
left_antshoulder_index = qualisys_marker_labels.index('L_AntShoulder')
left_postshoulder_index = qualisys_marker_labels.index('L_PostShoulder')

right_antshoulder_XYZ = qualisys_data[frame_to_use,right_antshoulder_index,:]
right_postshoulder_XYZ = qualisys_data[frame_to_use,right_postshoulder_index,:]
left_antshoulder_XYZ = qualisys_data[frame_to_use,left_antshoulder_index,:]
left_postshoulder_XYZ = qualisys_data[frame_to_use,left_postshoulder_index,:]

right_shoulder_XYZ = np.mean([right_antshoulder_XYZ,right_postshoulder_XYZ],axis=0)
left_shoulder_XYZ = np.mean([left_antshoulder_XYZ,left_postshoulder_XYZ],axis=0)

cspine_XYZ = np.mean([right_shoulder_XYZ,left_shoulder_XYZ],axis=0)

cspine_index = qualisys_joints.index('cspine')
qualisys_joints_array[frame_to_use,cspine_index,:] = cspine_XYZ



if debug:
        figure = plt.figure()
        ax1 = figure.add_subplot(121,projection = '3d')
        ax2 = figure.add_subplot(122, projection = '3d')

        ax1.scatter(qualisys_data[frame_to_use,:,0],qualisys_data[frame_to_use,:,1],qualisys_data[frame_to_use,:,2],c = 'r',marker = 'o')
        ax1.scatter(qualisys_joints_array[frame_to_use,:,0],qualisys_joints_array[frame_to_use,:,1],qualisys_joints_array[frame_to_use,:,2],c = 'b',marker = '.')
        set_axes_ranges(ax1,qualisys_data[frame_to_use,:,:],1000)

        ax2.scatter(qualisys_joints_array[frame_to_use,:,0],qualisys_joints_array[frame_to_use,:,1],qualisys_joints_array[frame_to_use,:,2],c = 'b',marker = 'o')
        set_axes_ranges(ax2,qualisys_joints_array[frame_to_use,:,:],1000)

        plt.show()





f=2