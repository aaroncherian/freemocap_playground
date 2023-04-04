from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from freemocap_utils import freemocap_data_loader
from freemocap_utils.mediapipe_skeleton_builder import mediapipe_indices
from scipy import signal


qualisys_indices = [
'head',
'left_ear',
'right_ear',
'cspine',
'left_shoulder',
'right_shoulder',
'left_elbow',
'right_elbow',
'left_wrist',
'right_wrist',
'left_index',
'right_index',
'left_hip',
'right_hip',
'left_knee',
'right_knee',
'left_ankle',
'right_ankle',
'left_heel',
'right_heel',
'left_foot_index',
'right_foot_index',
]



path_to_data_folder = Path(r'D:\ValidationStudy2022\FreeMocap_Data')

sessionID_list = ['sesh_2022-05-24_16_02_53_JSM_T1_NIH',]
#joint_to_plot = 'right_wrist'

labels = ['front_side_back', 'front', 'front_side', 'front_back']

#qualisys_joint_index = qualisys_indices.index(joint_to_plot)
path_to_qualysis_session_folder = Path(r"D:\ValidationStudy_numCams\FreeMoCap_Data\qualisys_sesh_2022-05-24_16_02_53_JSM_T1_WalkRun")
#qualisys_data = np.load(path_to_qualysis_session_folder/'DataArrays'/'qualisys_origin_aligned_skeleton_3D.npy')
qualisys_data = np.load(path_to_qualysis_session_folder/'DataArrays'/'totalBodyCOM_frame_XYZ.npy')
samples = qualisys_data.shape[0]

qualisys_sliced = qualisys_data[0:5243,:,:]

# q = 10

# samples_decimated = int(samples/q)
# this_joint_qual = qualisys_data[0:52430,qualisys_joint_index,:]

# qualisys_timeseries_x = signal.decimate(this_joint_qual[:,0],10)
# qualisys_timeseries_y = signal.decimate(this_joint_qual[:,1],10)
# qualisys_timeseries_z = signal.decimate(this_joint_qual[:,2],10)




figure = plt.figure()

x_ax = figure.add_subplot(311)
y_ax = figure.add_subplot(312)
z_ax = figure.add_subplot(313)

#create a dataholder instance for all sessions in the list
freemocap_sessions_dict = {}
for count,sessionID in enumerate(sessionID_list):
    freemocap_sessions_dict[count] = freemocap_data_loader.FreeMoCapDataLoader(path_to_data_folder/sessionID)

#create a dictionary of the joint XYZ data for the chosen joint
mediapipe_joint_data_dict = {}
com_data_dict = {}
for count,session_data in enumerate(freemocap_sessions_dict.values()):
    mediapipe_data = session_data.load_mediapipe_body_data()
    mediapipe_joint_data = mediapipe_data[1157:6400,mediapipe_indices.index(joint_to_plot),:]
    
    diff_to_zero = mediapipe_joint_data[0,:]

    
    #mediapipe_joint_data_dict[count] = mediapipe_joint_data-diff_to_zero
    mediapipe_joint_data_dict[count] = mediapipe_joint_data

    com_data = session_data.load_total_body_COM_data()
    com_data_dict[count] = com_data


for count,joint_data in enumerate(mediapipe_joint_data_dict.values()):
    x_ax.plot(joint_data[:,0], label = labels[count])
    y_ax.plot(joint_data[:,1], label = labels[count])
    z_ax.plot(np.abs(joint_data[:,2]), label = labels[count])

##to plot differences
# for count,joint_data in enumerate(mediapipe_joint_data_dict.values()):
#     x_ax.plot(joint_data[:,0] - qualisys_timeseries, label = labels[count])
#     y_ax.plot(joint_data[:,0], label = labels[count])
#     z_ax.plot(joint_data[:,2], label = labels[count])

#x_ax.plot(qualisys_timeseries_x-qualisys_timeseries_x[0],label = 'Qualisys')
x_ax.plot(qualisys_sliced[:,qualisys_joint_index,0])
y_ax.plot(qualisys_sliced[:,qualisys_joint_index,1])
# y_ax.plot(qualisys_timeseries_y)
#z_ax.plot(qualisys_timeseries_z)
#y_ax.plot(qualisys_timeseries,label = 'Qualisys')

x_ax.legend()
x_ax.set_ylabel('X Axis Position (mm)')
y_ax.set_ylabel('Y Axis Position (mm)')
z_ax.set_ylabel('Z Axis Position (mm)')
z_ax.set_xlabel('Frame #')
figure.suptitle(f'{joint_to_plot} Position Trajectory')

plt.show()

# for count,joint_data in enumerate(com_data_dict.values()):
#     x_ax.plot(joint_data[:,0], label = labels[count])
#     y_ax.plot(joint_data[:,1], label = labels[count])
#     z_ax.plot(joint_data[:,2], label = labels[count])

# #figure.suptitle('COM Trajectory')t


f = 2

