import pandas as pd
import numpy as np

from mediapipe_skeleton_builder import mediapipe_indices, build_mediapipe_skeleton
from COM_calculator import calculate_segment_COM, reformat_segment_COM, calculate_total_body_COM

def build_anthropometric_dataframe(segments:list,joint_connections:list,segment_COM_lengths:list,segment_COM_percentages:list) -> pd.DataFrame:
    #load anthropometric data into a pandas dataframe
    df = pd.DataFrame(list(zip(segments,joint_connections,segment_COM_lengths,segment_COM_percentages)),columns = ['Segment_Name','Joint_Connection','Segment_COM_Length','Segment_COM_Percentage'])
    segment_conn_len_perc_dataframe = df.set_index('Segment_Name')
    return segment_conn_len_perc_dataframe

def run(freemocap_marker_data_array:np.ndarray, pose_estimation_skeleton:list, anthropometric_info_dataframe:pd.DataFrame):

    num_frames = freemocap_marker_data_array.shape[0]
    num_frame_range = range(num_frames)
    num_segments = len(anthropometric_info_dataframe)

    segment_COM_frame_dict = calculate_segment_COM(anthropometric_info_dataframe, pose_estimation_skeleton, num_frame_range)
    segment_COM_frame_imgPoint_XYZ = reformat_segment_COM(segment_COM_frame_dict,num_frame_range, num_segments)
    totalBodyCOM_frame_XYZ = calculate_total_body_COM(anthropometric_info_dataframe,segment_COM_frame_dict,num_frame_range)

    return segment_COM_frame_dict,segment_COM_frame_imgPoint_XYZ,totalBodyCOM_frame_XYZ

if __name__ == '__main__':
    from pathlib import Path
    import numpy as np
     
    from anthropometry_data_tables import segments, joint_connections, segment_COM_lengths, segment_COM_percentages

    freemocap_data_folder_path = Path(r'D:\ValidationStudy2022\FreeMocap_Data')


    sessionID = 'sesh_2022-05-03_13_43_00_JSM_treadmill_day2_t0' #name of the sessionID folder

    data_array_path = freemocap_data_folder_path/sessionID/'DataArrays'
    freemocap_marker_data_array = np.load(data_array_path/'mediaPipeSkel_3d_smoothed.npy')

    anthropometric_info_dataframe = build_anthropometric_dataframe(segments,joint_connections,segment_COM_lengths,segment_COM_percentages)
    skelcoordinates_frame_segment_joint_XYZ = build_mediapipe_skeleton(freemocap_marker_data_array,anthropometric_info_dataframe,mediapipe_indices)
    segment_COM_frame_dict,segment_COM_frame_imgPoint_XYZ,totalBodyCOM_frame_XYZ = run(freemocap_marker_data_array,skelcoordinates_frame_segment_joint_XYZ, anthropometric_info_dataframe) 


f = 2