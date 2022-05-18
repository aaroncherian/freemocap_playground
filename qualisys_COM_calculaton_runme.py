import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path 
import socket
import pickle
import scipy.io as sio

from anthropometry_data_tables import segments, joint_connections, segment_COM_lengths, segment_COM_percentages
from qualisys_skeleton_builder import qualisys_indices, build_qualisys_skeleton
from COM_calculator import calculate_segment_COM_for_qualisys, reformat_segment_COM, calculate_total_body_COM

this_computer_name = socket.gethostname()
print(this_computer_name)

if this_computer_name == 'DESKTOP-V3D343U':
        freemocap_data_path = Path(r"I:\My Drive\HuMoN_Research_Lab\FreeMoCap_Stuff\FreeMoCap_Balance_Validation\data")
elif this_computer_name == 'DESKTOP-F5LCT4Q':
    freemocap_data_path = Path(r"C:\Users\aaron\Documents\HumonLab\Spring2022\ValidationStudy\FreeMocap_Data")
    #freemocap_data_path = Path(r'D:\freemocap2022\FreeMocap_Data')
else:
    #freemocap_validation_data_path = Path(r"C:\Users\kiley\Documents\HumonLab\SampleFMC_Data\FreeMocap_Data-20220216T173514Z-001\FreeMocap_Data")
    freemocap_data_path = Path(r"C:\Users\Rontc\Documents\HumonLab\ValidationStudy")

sessionID = 'session_SER_1_20_22' #name of the sessionID folder
data_array_name = 'skeleton_fr_mar_dim_rotated.mat'

this_freemocap_session_path = freemocap_data_path / sessionID
this_freemocap_data_path = this_freemocap_session_path/'DataArrays'
qualisys_data_path = this_freemocap_data_path/data_array_name

qualisys_skeleton_file_path = this_freemocap_data_path/'qualisys_Skelcoordinates_frame_segment_joint_XYZ.pkl'
segmentCOM_data_path = this_freemocap_data_path/'qualisys_segmentedCOM_frame_joint_XYZ.npy'
totalBodyCOM_data_path = this_freemocap_data_path/'qualisys_totalBodyCOM_frame_XYZ.npy'

qualysis_mat_file = sio.loadmat(qualisys_data_path)

qualisys_pose_data = qualysis_mat_file['skeleton_fr_mar_dim_rotated']

num_frame_range = range(qualisys_pose_data.shape[0])

df = pd.DataFrame(list(zip(segments,joint_connections,segment_COM_lengths,segment_COM_percentages)),columns = ['Segment Name','Joint Connection','Segment COM Length','Segment COM Percentage'])
segment_conn_len_perc_dataframe = df.set_index('Segment Name')
num_segments = len(segments)

skelcoordinates_frame_segment_joint_XYZ = build_qualisys_skeleton(qualisys_pose_data,segment_conn_len_perc_dataframe, qualisys_indices, num_frame_range)

segment_COM_frame_dict = calculate_segment_COM_for_qualisys(segment_conn_len_perc_dataframe, skelcoordinates_frame_segment_joint_XYZ, num_frame_range)
segment_COM_frame_imgPoint_XYZ = reformat_segment_COM(segment_COM_frame_dict,num_frame_range, num_segments)
totalBodyCOM_frame_XYZ = calculate_total_body_COM(segment_conn_len_perc_dataframe,segment_COM_frame_dict,num_frame_range)

#save out files 
open_file = open(qualisys_skeleton_file_path, "wb")
pickle.dump(skelcoordinates_frame_segment_joint_XYZ, open_file)
open_file.close()

np.save(segmentCOM_data_path,segment_COM_frame_imgPoint_XYZ)

np.save(totalBodyCOM_data_path,totalBodyCOM_frame_XYZ,num_frame_range)
