import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path 
import socket
import pickle
import scipy.io as sio

from fmc_qual_validation_toolbox.anthropometry_data_tables import segments, joint_connections, segment_COM_lengths, segment_COM_percentages
from fmc_qual_validation_toolbox.mediapipe_skeleton_builder import mediapipe_indices, build_mediapipe_skeleton, slice_mediapipe_data
from fmc_qual_validation_toolbox.COM_calculator import calculate_segment_COM,calculate_segment_COM_for_qualisys, reformat_segment_COM, calculate_total_body_COM
from fmc_qual_validation_toolbox.qualisys_skeleton_builder import qualisys_indices, build_qualisys_skeleton



def run(session_info, freemocap_data_array_path, skeleton_data):

    sessionID = session_info['sessionID']
    skeleton_type = session_info['skeleton_type']

    #data_array_name = '{}_origin_aligned_skeleton_3D.npy'.format(skeleton_type)

    num_frames = skeleton_data.shape[0]
    num_frame_range = range(num_frames)
    #creating paths to the session and data
    #this_freemocap_session_path = freemocap_data_folder_path / sessionID
    #this_freemocap_data_path = this_freemocap_session_path/'DataArrays'
    #data_array_path = this_freemocap_data_path/data_array_name


    skeleton_file_path = Path(r'D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_14_40_56_MDN_NIH_Trial2\output_data\center_of_mass\origin_aligned_{}_Skelcoordinates_frame_segment_joint_XYZ.pkl'.format(skeleton_type))
    segmentCOM_data_path = Path(r'D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_14_40_56_MDN_NIH_Trial2\output_data\center_of_mass\origin_aligned_segmentedCOM_frame_joint_XYZ.npy')
    totalBodyCOM_data_path = freemocap_data_array_path/'output_data'/'center_of_mass'/'total_body_center_of_mass_xyz.npy'



    #load the mediapipe data
    # if skeleton_type == 'mediapipe':
    #     skeleton_data_all_joints = np.load(data_array_path)
    #     num_pose_joints = 33 
    #     #get just the body data for mediapipe 
    #     skeleton_data = slice_mediapipe_data(skeleton_data_all_joints, num_pose_joints)
    #     num_frame_range = range(len(skeleton_data))

    # elif skeleton_type == 'qualisys':
    #     skeleton_data = np.load(data_array_path)
    #     num_frames = skeleton_data.shape[0]
    #     num_frame_range = range(num_frames)

    #load anthropometric data into a pandas dataframe
    df = pd.DataFrame(list(zip(segments,joint_connections,segment_COM_lengths,segment_COM_percentages)),columns = ['Segment Name','Joint Connection','Segment COM Length','Segment COM Percentage'])
    segment_conn_len_perc_dataframe = df.set_index('Segment Name')
    num_segments = len(segments)

    #build a mediapipe skeleton based on the segments defined in the anthropometry_data_tables.py file
    if skeleton_type == 'mediapipe':
        skelcoordinates_frame_segment_joint_XYZ = build_mediapipe_skeleton(skeleton_data,segment_conn_len_perc_dataframe, mediapipe_indices, num_frame_range)
        #segment_COM_frame_dict = calculate_segment_COM(segment_conn_len_perc_dataframe, skelcoordinates_frame_segment_joint_XYZ, num_frame_range)
    elif skeleton_type == 'qualisys':
        skelcoordinates_frame_segment_joint_XYZ = build_qualisys_skeleton(skeleton_data,segment_conn_len_perc_dataframe, qualisys_indices, num_frame_range)
        #segment_COM_frame_dict = calculate_segment_COM(segment_conn_len_perc_dataframe, skelcoordinates_frame_segment_joint_XYZ, num_frame_range)
        #segment_COM_frame_dict = calculate_segment_COM_for_qualisys(segment_conn_len_perc_dataframe, skelcoordinates_frame_segment_joint_XYZ, num_frame_range)
    #calculate segment and total body COM data 

    # segment_COM_frame_dict = calculate_segment_COM(segment_conn_len_perc_dataframe, skelcoordinates_frame_segment_joint_XYZ, num_frame_range)
    # segment_COM_frame_imgPoint_XYZ = reformat_segment_COM(segment_COM_frame_dict,num_frame_range, num_segments)
    # totalBodyCOM_frame_XYZ = calculate_total_body_COM(segment_conn_len_perc_dataframe,segment_COM_frame_dict,num_frame_range)

    #save out files 
    open_file = open(skeleton_file_path, "wb")
    pickle.dump(skelcoordinates_frame_segment_joint_XYZ, open_file)
    open_file.close()

    # np.save(segmentCOM_data_path,segment_COM_frame_imgPoint_XYZ)

    # np.save(totalBodyCOM_data_path,totalBodyCOM_frame_XYZ,num_frame_range)

if __name__ == '__main__':

    this_computer_name = socket.gethostname()
    print(this_computer_name)

    if this_computer_name == 'DESKTOP-V3D343U':
            freemocap_data_path = Path(r"I:\My Drive\HuMoN_Research_Lab\FreeMoCap_Stuff\FreeMoCap_Balance_Validation\data")
    elif this_computer_name == 'DESKTOP-F5LCT4Q':
        #freemocap_data_path = Path(r"C:\Users\aaron\Documents\HumonLab\Spring2022\ValidationStudy\FreeMocap_Data")
        #freemocap_data_path = Path(r'D:\freemocap2022\FreeMocap_Data')
        freemocap_data_folder_path = Path(r'D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3')
    else:
        #freemocap_validation_data_path = Path(r"C:\Users\kiley\Documents\HumonLab\SampleFMC_Data\FreeMocap_Data-20220216T173514Z-001\FreeMocap_Data")
        freemocap_data_path = Path(r"C:\Users\Rontc\Documents\HumonLab\ValidationStudy")

    #sessionID = 'sesh_2022-05-03_13_43_00_JSM_treadmill_day2_t0' #name of the sessionID folder
    #sessionID = 'gopro_sesh_2022-05-24_16_02_53_JSM_T1_NIH'

    session_info = {'sessionID': 'sesh_2023-05-17_14_40_56_MDN_NIH_Trial2', 'skeleton_type': 'mediapipe'}
    

    freemocap_data_array_path = freemocap_data_folder_path/session_info['sessionID']/'output_data'/'mediapipe_body_3d_xyz.npy'
    skeleton_data = np.load(freemocap_data_array_path)

    run(session_info, freemocap_data_folder_path,skeleton_data) 


f = 2