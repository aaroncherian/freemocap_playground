import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path 
import socket
import pickle
from tqdm import tqdm
import cv2
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import sys 

from io import BytesIO


#you can skip every 10th frame 

this_computer_name = socket.gethostname()
print(this_computer_name)


if this_computer_name == 'DESKTOP-V3D343U':
    freemocap_validation_data_path = Path(r"I:\My Drive\HuMoN_Research_Lab\FreeMoCap_Stuff\FreeMoCap_Balance_Validation\data")
elif this_computer_name == 'DESKTOP-F5LCT4Q':
    #freemocap_validation_data_path = Path(r"C:\Users\aaron\Documents\HumonLab\Spring2022\ValidationStudy\FreeMocap_Data")
    freemocap_validation_data_path = Path(r'D:\freemocap2022\FreeMocap_Data')
else:
    #freemocap_validation_data_path = Path(r"C:\Users\kiley\Documents\HumonLab\SampleFMC_Data\FreeMocap_Data-20220216T173514Z-001\FreeMocap_Data")
    freemocap_validation_data_path = Path(r"C:\Users\Rontc\Documents\HumonLab\ValidationStudy")
sessionID = 'sesh_2022-02-25_17_57_39' #name of the sessionID folder
this_freemocap_session_path = freemocap_validation_data_path / sessionID
this_freemocap_data_path = this_freemocap_session_path/'DataArrays'

syncedVideoName = sessionID + '_Cam1_synced.mp4'

syncedVideoPath = this_freemocap_session_path/'SyncedVideos'/syncedVideoName

mediapipe_data_path = this_freemocap_data_path/'mediaPipeSkel_3d.npy'

mediapipeSkel_fr_mar_dim = np.load(mediapipe_data_path) #loads in the data as a numpy array

num_pose_joints = 33 #number of pose joints tracked by mediapipe 
pose_joint_range = range(num_pose_joints)

mediapipeSkel_fr_mar_dim = np.load(mediapipe_data_path)

mediapipe_pose_data = mediapipeSkel_fr_mar_dim[:,0:33,:]




skel_x = mediapipe_pose_data[:,:,0]
skel_y = mediapipe_pose_data[:,:,1]
skel_z = mediapipe_pose_data[:,:,2]


figure = plt.figure()
ax = figure.add_subplot( projection = '3d')

frame_to_plot = 300

this_frame_skel_x = skel_x[frame_to_plot,:]
this_frame_skel_y = skel_y[frame_to_plot,:]
this_frame_skel_z = skel_z[frame_to_plot,:]

ax.scatter(this_frame_skel_x,this_frame_skel_y,this_frame_skel_z, color = 'grey')


left_foot_index = 32
right_foot_index = 31

ax.scatter(this_frame_skel_x[left_foot_index],this_frame_skel_y[left_foot_index],this_frame_skel_z[left_foot_index],color = 'blue')
ax.scatter(this_frame_skel_x[right_foot_index],this_frame_skel_y[right_foot_index],this_frame_skel_z[right_foot_index],color = 'red')
plt.show()
f=2