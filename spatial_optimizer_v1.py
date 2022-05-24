from distutils.log import debug
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
from scipy import optimize
from scipy.spatial.transform import Rotation
import scipy.io as sio

from mediapipe_skeleton_builder import mediapipe_indices, build_mediapipe_skeleton, slice_mediapipe_data
from anthropometry_data_tables import segments, joint_connections, segment_COM_lengths, segment_COM_percentages
from qualisys_skeleton_builder import qualisys_indices, build_qualisys_skeleton

this_computer_name = socket.gethostname()

if this_computer_name == 'DESKTOP-V3D343U':
    freemocap_data_path = Path(r"I:\My Drive\HuMoN_Research_Lab\FreeMoCap_Stuff\FreeMoCap_Balance_Validation\data")
elif this_computer_name == 'DESKTOP-F5LCT4Q':
    freemocap_data_path = Path(r"C:\Users\aaron\Documents\HumonLab\Spring2022\ValidationStudy\FreeMocap_Data")
    #freemocap_data_path = Path(r'D:\freemocap2022\FreeMocap_Data')
else:
    #freemocap_validation_data_path = Path(r"C:\Users\kiley\Documents\HumonLab\SampleFMC_Data\FreeMocap_Data-20220216T173514Z-001\FreeMocap_Data")
    freemocap_data_path = Path(r"C:\Users\Rontc\Documents\HumonLab\ValidationStudy")


sessionID = 'session_SER_1_20_22' #name of the sessionID folder

qualisys_data_array_name = 'skeleton_fr_mar_dim_rotated.mat'

mediapipe_data_array_name = 'mediapipe_origin_aligned_skeleton_3D.npy'
num_pose_joints = 33

this_freemocap_session_path = freemocap_data_path / sessionID
this_freemocap_data_path = this_freemocap_session_path/'DataArrays'
qualisys_data_path = this_freemocap_data_path/qualisys_data_array_name
mediapipe_data_path = this_freemocap_data_path/mediapipe_data_array_name

qualysis_mat_file = sio.loadmat(qualisys_data_path)
qualisys_pose_data = qualysis_mat_file['skeleton_fr_mar_dim_rotated']
#qualisys_num_frame_range = range(qualisys_pose_data.shape[0])
qualisys_num_frame_range = range(20000)

mediapipeSkel_fr_mar_dim = np.load(mediapipe_data_path)
mediapipe_pose_data = slice_mediapipe_data(mediapipeSkel_fr_mar_dim, num_pose_joints)
mediapipe_num_frame_range = range(len(mediapipe_pose_data))

mediapipe_skeleton_path = this_freemocap_data_path/'origin_aligned_mediapipeSkelcoordinates_frame_segment_joint_XYZ.pkl'
qualisys_skeleton_path = this_freemocap_data_path/'qualisys_Skelcoordinates_frame_segment_joint_XYZ.pkl'

df = pd.DataFrame(list(zip(segments,joint_connections,segment_COM_lengths,segment_COM_percentages)),columns = ['Segment Name','Joint Connection','Segment COM Length','Segment COM Percentage'])
segment_conn_len_perc_dataframe = df.set_index('Segment Name')

if mediapipe_skeleton_path.is_file():
    with open(mediapipe_skeleton_path, 'rb') as f:
        mediapipe_skeleton_data = pickle.load(f)
    f.close()
else:
    mediapipe_skeleton_data = build_mediapipe_skeleton(mediapipe_pose_data,segment_conn_len_perc_dataframe, mediapipe_indices, mediapipe_num_frame_range)

if qualisys_skeleton_path.is_file():
    with open(qualisys_skeleton_path, 'rb') as f:
        qualisys_skeleton_data = pickle.load(f)
    f.close()
else:
    qualisys_skeleton_data = build_qualisys_skeleton(qualisys_pose_data,segment_conn_len_perc_dataframe, qualisys_indices, qualisys_num_frame_range)






f=2