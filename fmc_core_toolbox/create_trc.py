import csv


def save_trajectory_trc(skeleton_data_frame, keypoints_names, frame_rate, trc_save_path):
    
    data_rate = camera_rate = orig_data_rate = frame_rate
    num_frames = len(skeleton_data_frame)
    num_frame_range = range(num_frames)
    num_markers = len(keypoints_names)
    units = 'mm'
    orig_data_start_frame = num_frame_range[0]
    orig_num_frames = num_frame_range[-1]
    
    trc_filename = trc_save_path.parts[-1]

    with open(trc_save_path, 'wt', newline='', encoding='utf-8') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(["PathFileType",
                            "4", 
                            "(X/Y/Z)",	
                            trc_filename])
        tsv_writer.writerow(["DataRate",
                            "CameraRate",
                            "NumFrames",
                            "NumMarkers", 
                            "Units",
                            "OrigDataRate",
                            "OrigDataStartFrame",
                            "OrigNumFrames"])
        tsv_writer.writerow([data_rate, 
                            camera_rate,
                            num_frames, 
                            num_markers, 
                            units, 
                            orig_data_rate, 
                            orig_data_start_frame, 
                            orig_num_frames])

        header_names = ['Frame#', 'Time']
        for keypoint in keypoints_names:
            header_names.append(keypoint)
            header_names.append("")
            header_names.append("")

        tsv_writer.writerow(header_names)

        header_names = ["",""]
        for i in range(1,len(keypoints_names)+1):
            header_names.append("X"+str(i))
            header_names.append("Y"+str(i))
            header_names.append("Z"+str(i))    
        
        tsv_writer.writerow(header_names)
        tsv_writer.writerow("")    

        skeleton_data_frame.insert(0, "Frame", [str(i) for i in range(0, len(skeleton_data_frame))])
        skeleton_data_frame.insert(1, "Time", skeleton_data_frame["Frame"].astype(float) / float(camera_rate))

        for row in range(0, len(skeleton_data_frame)):
            tsv_writer.writerow(skeleton_data_frame.iloc[row].tolist())

        f = 2 


def flatten_freemocap_data(freemocap_marker_data):
    num_frames = freemocap_marker_data.shape[0]
    num_markers = freemocap_marker_data.shape[1]

    skeleton_data_flat = freemocap_marker_data.reshape(num_frames,num_markers*3)

    return skeleton_data_flat


if __name__ == '__main__':

  
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from mediapipe_skeleton_builder import mediapipe_indices, slice_mediapipe_data   

    freemocap_data_folder_path = Path(r'D:\ValidationStudy2022\FreeMocap_Data')


    sessionID = 'sesh_2022-05-24_15_55_40_JSM_T1_BOS'
    data_array_folder = 'DataArrays'
    array_name = 'mediaPipeSkel_3d_filtered.npy'
    trc_save_name = 'trc_test.trc'


    data_array_folder_path = freemocap_data_folder_path / sessionID / data_array_folder
    freemocap_marker_data = np.load(data_array_folder_path / array_name)
    trc_save_path = data_array_folder_path/trc_save_name

    freemocap_body_marker_data = slice_mediapipe_data(freemocap_marker_data,len(mediapipe_indices))
    freemocap_flat_body_marker_data = flatten_freemocap_data(freemocap_body_marker_data)

    frame_rate = 30

    freemocap_marker_data_frame = pd.DataFrame(freemocap_flat_body_marker_data)

    save_trajectory_trc(freemocap_marker_data_frame,mediapipe_indices, frame_rate, trc_save_path)


