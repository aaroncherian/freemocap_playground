from pathlib import Path

from freemocap_utils.mediapipe_skeleton_builder import mediapipe_indices
from freemocap_utils import freemocap_data_loader
import sys

import numpy as np

import pandas as pd


import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error

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

def calculate_rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


path_to_data_folder = Path(r'D:\ValidationStudy_numCams\FreeMoCap_Data')

sessionID_list = ['sesh_2022-05-24_16_10_46_WalkRun_front','sesh_2022-05-24_16_10_46_WalkRun_front_side','sesh_2022-05-24_16_10_46_WalkRun_front_back','sesh_2022-05-24_16_10_46_JSM_T1_WalkRun']
labels = ['front', 'front_side','front_back','front_side_back']

# sessionID_list = ['sesh_2022-05-24_16_10_46_JSM_T1_WalkRun']
# labels = ['front_side_back']



# path_to_data_folder = Path(r'D:\ValidationStudy2022\FreeMoCap_Data')
# sessionID_list = ['sesh_2022-05-24_15_55_40_JSM_T1_BOS']
# labels = ['FreeMoCap']
# sessionID_list = ['sesh_2022-05-24_16_10_46_WalkRun_front','sesh_2022-05-24_16_10_46_WalkRun_front_side','sesh_2022-05-24_16_10_46_JSM_T1_WalkRun']
# labels = ['front', 'front_side','front_side_back']

#path_to_qualysis_session_folder = Path(r"D:\ValidationStudy_numCams\FreeMoCap_Data\qualisys_sesh_2022-05-24_16_02_53_JSM_T1_WalkRun")
#qualisys_data = np.load(path_to_qualysis_session_folder/'DataArrays'/'qualisys_origin_aligned_skeleton_3D.npy')
path_to_qualisys_session_folder = Path(r"D:\ValidationStudy2022\FreeMocap_Data\qualisys_sesh_2022-05-24_16_02_53_JSM_T1_WalkRun")
qualisys_data = np.load(path_to_qualisys_session_folder/'DataArrays'/'qualisys_marker_data_29Hz.npy')
samples = qualisys_data.shape[0]


frame_range = None

if frame_range:
    qualisys_sliced = qualisys_data[frame_range[0]:frame_range[1],:,:] - qualisys_data[frame_range[0],:,:]
else:
    qualisys_sliced = qualisys_data[:,:,:] - qualisys_data[0,:,:]


freemocap_sessions_dict = {}
for count,sessionID in enumerate(sessionID_list):
    freemocap_sessions_dict[count] = freemocap_data_loader.FreeMoCapDataLoader(path_to_data_folder/sessionID)

mediapipe_joint_data_dict = {}

for count,session_data in enumerate(freemocap_sessions_dict.values()):
    mediapipe_data = session_data.load_mediapipe_body_data() 
    mediapipe_joint_data = mediapipe_data[1162:6621,:,:]
    if frame_range:
        mediapipe_joint_data_dict[count] = mediapipe_joint_data[frame_range[0]:frame_range[1],:,:] - mediapipe_joint_data[frame_range[0],:,:] 
    else:
        mediapipe_joint_data_dict[count] = mediapipe_joint_data[:,:,:] - mediapipe_joint_data[0,:,:] 


rmse_dict = {}

rmse_dataframe = pd.DataFrame()

dimension_list = ['x','y','z']
session_dict = {}

for marker in mediapipe_indices:
    if marker in qualisys_indices: #only run rmse if the marker is in both qualisys and freemocap
        mediapipe_marker = mediapipe_indices.index(marker)
        qualisys_marker = qualisys_indices.index(marker)

        for session_count, mediapipe_session in enumerate(mediapipe_joint_data_dict.values()): #for each of the sessions, calculate the rmse per dimension for each marker
            rmse_list = [] 
            for dimension_count,dimension in enumerate(dimension_list):
                rmse = calculate_rmse(qualisys_sliced[:,qualisys_marker,dimension_count],mediapipe_session[:,mediapipe_marker,dimension_count])


                rmse_list.append(rmse) #add rmse per dimension to a list 

            rmse_dict = {'marker': marker, 'rmse': rmse_list, 'dimension': dimension_list, 'session':labels[session_count]}
            rmse_pd = pd.DataFrame(rmse_dict)
            rmse_dataframe = pd.concat([rmse_pd,rmse_dataframe])
            f =2 
        

        f = 2


sns.set_theme(style="whitegrid")

for dimension in dimension_list:
    ax = sns.barplot(
        data= rmse_dataframe.loc[rmse_dataframe['dimension'] == dimension],
        x= "marker", y="rmse", hue="session",
        errorbar="sd", palette="dark", alpha = .6,
    )

    ax.tick_params(axis = 'x', rotation = 90)

    fig = ax.get_figure()
    plt.show()


f =2 



