from pathlib import Path

import matplotlib.pyplot as plt

from freemocap_utils import freemocap_data_loader
from freemocap_utils.mediapipe_skeleton_builder import mediapipe_indices

path_to_data_folder = Path(r'D:\ValidationStudy_numCams\FreeMoCap_Data')

sessionID_list = ['sesh_2022-05-24_16_10_46_JSM_T1_WalkRun','sesh_2022-05-24_16_10_46_WalkRun_front','sesh_2022-05-24_16_10_46_WalkRun_front_side']
joint_to_plot = 'right_ankle'

labels = ['front_side_back', 'front', 'front_side']


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
    mediapipe_joint_data = mediapipe_data[:,mediapipe_indices.index(joint_to_plot),:]
    mediapipe_joint_data_dict[count] = mediapipe_joint_data

    com_data = session_data.load_total_body_COM_data()
    com_data_dict[count] = com_data


for count,joint_data in enumerate(mediapipe_joint_data_dict.values()):
    x_ax.plot(joint_data[:,0], label = labels[count])
    y_ax.plot(joint_data[:,1], label = labels[count])
    z_ax.plot(joint_data[:,2], label = labels[count])

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

