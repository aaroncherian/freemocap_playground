

from imageio import save


def create_trc_from_numpy(skeleton_3d_data, trc_file_name, frame_rate, mediapipe_indices):
    
    NumFrames = skeleton_3d_data.shape[0]
    NumMarkers = (skeleton_3d_data.shape[1]-1)/3
    num_frame_range = range(NumFrames)

    DataRate = CameraRate = OrigDataRate = frame_rate

    #skeleton_3d_data.index = np.array(num_frame_range) + 1
    #skeleton_3d_data.insert(0, 't', skeleton_3d_data.index / frame_rate)


    header_trc = ['PathFileType\t4\t(X/Y/Z)\t' + trc_file_name, 
        'DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames', 
        '\t'.join(map(str,[DataRate, CameraRate, NumFrames, NumMarkers, 'mm', OrigDataRate, num_frame_range[0], num_frame_range[-1]])),
        'Frame#\tTime\t' + '\t\t\t'.join(mediapipe_indices) + '\t\t',
        '\t\t'+'\t'.join([f'X{i+1}\tY{i+1}\tZ{i+1}' for i in range(len(mediapipe_indices))])]


    return header_trc, skeleton_3d_data

def flatten_mediapipe_data(skeleton_3d_data):
    num_frames = skeleton_3d_data.shape[0]
    num_markers = skeleton_3d_data.shape[1]

    skeleton_data_flat = skeleton_3d_data.reshape(num_frames,num_markers*3)

    return skeleton_data_flat

def create_dataframe(skeleton_data_flat, frame_rate):
    num_frames = skeleton_data_flat.shape[0]

    timestamp_array = np.divide(np.arange(0,num_frames),frame_rate)

    #skeleton_data_flat = np.divide(skeleton_data_flat,1000)

    combined_array = np.column_stack((timestamp_array,skeleton_data_flat),)

    

    skel_dataframe = pd.DataFrame(combined_array)

    return skel_dataframe

    f = 2


def make_trc(Q, keypoints_names, f_range, data_array_folder_path):
    '''
    Make Opensim compatible trc file from a dataframe with 3D coordinates

    INPUT:
    - config: dictionary of configuration parameters
    - Q: pandas dataframe with 3D coordinates as columns, frame number as rows
    - keypoints_names: list of strings
    - f_range: list of two numbers. Range of frames

    OUTPUT:
    - trc file
    '''

    # Read config

    trc_f = f'{keypoints_names[0]}_{f_range[0]}-{f_range[1]}.trc'
    frame_rate = 30
    #Header
    DataRate = CameraRate = OrigDataRate = frame_rate
    NumFrames = len(Q)
    NumMarkers = len(keypoints_names)
    
    header_trc = ['PathFileType\t4\t(X/Y/Z)\t' + trc_f, 
            'DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames', 
            '\t'.join(map(str,[DataRate, CameraRate, NumFrames, NumMarkers, 'm', OrigDataRate, f_range[0], f_range[1]])),
            'Frame#\tTime\t' + '\t\t\t'.join(keypoints_names) + '\t\t',
            '\t\t'+'\t'.join([f'X{i+1}\tY{i+1}\tZ{i+1}' for i in range(len(keypoints_names))])]
    
    # Zup to Yup coordinate system
    #Q = zup2yup(Q)
    
    #Add Frame# and Time columns
    Q.index = np.array(range(f_range[0], f_range[1])) + 1
    Q.insert(0, 't', Q.index / frame_rate)

    #Write file
    trc_path = data_array_folder_path/trc_f
    with open(trc_path, 'w') as trc_o:
        [trc_o.write(line+'\n') for line in header_trc]
        Q.to_csv(trc_o, sep='\t', index=True, header=None, line_terminator='\n')

    return trc_path



if __name__ == '__main__':

    import socket
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from fmc_validation_toolbox.mediapipe_skeleton_builder import mediapipe_indices, slice_mediapipe_data   

    this_computer_name = socket.gethostname()

    if this_computer_name == 'DESKTOP-F5LCT4Q':
        #freemocap_validation_data_path = Path(r"C:\Users\aaron\Documents\HumonLab\Spring2022\ValidationStudy\FreeMocap_Data")
        #freemocap_data_folder_path = Path(r'D:\freemocap2022\FreeMocap_Data')
        freemocap_data_folder_path = Path(r'D:\ValidationStudy2022\FreeMocap_Data')
    else:
        freemocap_data_folder_path = Path(r'C:\Users\Aaron\Documents\sessions\FreeMocap_Data')

    #sessionID = 'sesh_2022-05-12_15_13_02'  
    #sessionID = 'sesh_2022-06-28_12_55_34'

    sessionID = 'sesh_2022-05-24_15_55_40_JSM_T1_BOS'
    data_array_folder = 'DataArrays'
    array_name = 'mediaPipeSkel_3d_filtered.npy'
    save_name = 'trc_test.trc'


    data_array_folder_path = freemocap_data_folder_path / sessionID / data_array_folder
    skel3d_data = np.load(data_array_folder_path / array_name)
    save_path = data_array_folder_path/save_name

    skel_3d_data = skel3d_data[0:5]
    skel_body_points = slice_mediapipe_data(skel3d_data,2)
    skel_3d_flat = flatten_mediapipe_data(skel_body_points)

    Q = pd.DataFrame(skel_3d_flat)

   

    #make_trc(Q, mediapipe_indices, [0,len(Q)], data_array_folder_path)
    #create_dataframe(skel_3d_flat,30)
    skel_3d_flat_dataframe = pd.DataFrame(skel_3d_flat)

    skel_3d_flat_dataframe = create_dataframe(skel_3d_flat,30)
    trc, skeleton_data_frame = create_trc_from_numpy(skel_3d_flat_dataframe, 'skel_trace_file.trc',30, mediapipe_indices)

    with open(save_path,'w') as trc_o:
        [trc_o.write(line+'\n') for line in trc]
        skeleton_data_frame.to_csv(trc_o, sep='\t', index=True, header=None, line_terminator='\n')

    # trc_o.close()
    

    f = 2
