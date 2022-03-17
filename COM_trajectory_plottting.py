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


this_computer_name = socket.gethostname()
print(this_computer_name)


if this_computer_name == 'DESKTOP-V3D343U':
    freemocap_validation_data_path = Path(r"I:\My Drive\HuMoN_Research_Lab\FreeMoCap_Stuff\FreeMoCap_Balance_Validation\data")
elif this_computer_name == 'DESKTOP-F5LCT4Q':
    freemocap_validation_data_path = Path(r"C:\Users\aaron\Documents\HumonLab\Spring2022\ValidationStudy\FreeMocap_Data")
    #freemocap_validation_data_path = Path(r'D:\freemocap2022\FreeMocap_Data')
else:
    freemocap_validation_data_path = Path(r"C:\Users\kiley\Documents\HumonLab\SampleFMC_Data\FreeMocap_Data-20220216T173514Z-001\FreeMocap_Data")


sessionID = 'session_SER_1_20_22' #name of the sessionID folder
this_freemocap_session_path = freemocap_validation_data_path / sessionID
this_freemocap_data_path = this_freemocap_session_path/'DataArrays'

totalCOM_data_path = this_freemocap_data_path / 'totalBodyCOM_frame_XYZ.npy'

#mediapipe_data_path = this_freemocap_data_path/'mediaPipeSkel_3d_smoothed.npy'
mediapipe_data_path = this_freemocap_data_path/'rotated_mediaPipeSkel_3d_smoothed.npy'



totalCOM_frame_XYZ = np.load(totalCOM_data_path) #loads in the data as a numpy array
mediapipeSkel_fr_mar_dim = np.load(mediapipe_data_path)

num_pose_joints = 33 #number of pose joints tracked by mediapipe 
mediapipe_pose_data = mediapipeSkel_fr_mar_dim[:,0:num_pose_joints,:] #load just the pose joints into a data array, removing hands and face data 

num_frame_range = range(9900,12000)


this_range_mediapipeSkel = mediapipe_pose_data[num_frame_range,:,:]

left_heel_index = 30
left_toe_index = 32

right_heel_index = 29
right_toe_index = 31


left_heel_x = this_range_mediapipeSkel[0,left_heel_index,[0]]
left_heel_z = this_range_mediapipeSkel[0,left_heel_index,[1]]


left_toe_x = this_range_mediapipeSkel[0,left_toe_index,[0]]
left_toe_z = this_range_mediapipeSkel[0,left_toe_index,[1]]

left_foot_x,left_foot_z = [left_heel_x,left_toe_x], [left_heel_z,left_toe_z]


right_heel_x = this_range_mediapipeSkel[0,right_heel_index,[0]]
right_heel_z = this_range_mediapipeSkel[0,right_heel_index,[1]]


right_toe_x = this_range_mediapipeSkel[0,right_toe_index,[0]]
right_toe_z = this_range_mediapipeSkel[0,right_toe_index,[1]]

right_foot_x,right_foot_z = [right_heel_x,right_toe_x], [right_heel_z,right_toe_z]



#left_foot = [this_range_mediapipeSkel[:,left_heel_index,[0,2]],this_range_mediapipeSkel[:,left_toe_index,[0,2]]]

#mean_left_foot = np.mean(left_foot,1)



this_range_totalCOM = totalCOM_frame_XYZ[num_frame_range,:]

figure = plt.figure()
ax = figure.add_subplot(111)
ax.scatter(this_range_totalCOM[:,0],this_range_totalCOM[:,1])
#ax.plot(mean_left_foot[:,0],mean_left_foot[:,1])
ax.plot(left_foot_x,left_foot_z, color = 'red')
ax.plot(right_foot_x,right_foot_z, color = 'blue')
plt.show()


f = 2