import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path 
import socket
import pickle
import scipy.io as sio


this_computer_name = socket.gethostname()

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
mediapipe_data_path = this_freemocap_data_path / 'mediaPipeSkel_3d_smoothed.npy'
qualisys_data_path = this_freemocap_data_path/'qualisysData_3d.mat'

qualysis_mat_file = sio.loadmat(qualisys_data_path)

qualisysSkel_fr_mar_dim = qualysis_mat_file['skeleton_fr_mar_dim_reorg'] 


skel_x = qualisysSkel_fr_mar_dim[:,:,0]
skel_y = qualisysSkel_fr_mar_dim[:,:,1]
skel_z = qualisysSkel_fr_mar_dim[:,:,2]

num_frames = qualisysSkel_fr_mar_dim.shape[0]

mx = np.nanmean(skel_x[int(num_frames/2),:])
my = np.nanmean(skel_y[int(num_frames/2),:])
mz = np.nanmean(skel_z[int(num_frames/2),:])




plt.ion
figure = plt.figure()
ax = figure.add_subplot(111, projection = '3d')

this_frame_qualisys_skel = qualisysSkel_fr_mar_dim[70000,:,:]

ind = [0,6,8]
points_of_interest = this_frame_qualisys_skel[ind,:]

ax_range = 1000
ax.scatter(points_of_interest[:,0],points_of_interest[:,1],points_of_interest[:,2], color = 'red')
ax.scatter(this_frame_qualisys_skel[:,0],this_frame_qualisys_skel[:,1],this_frame_qualisys_skel[:,2])

ax.set_xlim([mx-ax_range, mx+ax_range])
ax.set_ylim([my-ax_range, my+ax_range])
ax.set_zlim([mz-ax_range, mz+ax_range])
plt.show()
