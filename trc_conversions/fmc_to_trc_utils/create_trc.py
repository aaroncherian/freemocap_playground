import csv


def get_trajectory_dataframe(self):
    """returns a dataframe of trajectories for a session"""

    #TODO: resolve this via a good charuco board, not this hack
    scale_factor = 0.003

    # These will manipulate the orientation of the markers 
    # to make them line up better by default with OpenSim

    # flip y and z (1 and 2); y becomes negative
    axes_order= [0,1,2]
    axes_flip=[1,-1,-1]

    # Order of the Axes
    x_axis = axes_order[0]
    y_axis = axes_order[1]
    z_axis = axes_order[2]

    # Adjust axes to have alignment with the vertical
    flip_x = axes_flip[0]
    flip_y = axes_flip[1]
    flip_z = axes_flip[2]

    all_trajectories = self.get_trajectory_array()

    # not interested in face mesh or hands here, 
    # so only taking elements listed in landmarks.json
    # for the current mediapipe pose, just represents the gross pose + hands
    lm_x = (all_trajectories[:, 0:self.tracked_landmark_count, x_axis] * flip_x * scale_factor)   # skeleton x data
    lm_y = (all_trajectories[:, 0:self.tracked_landmark_count, y_axis] * flip_y * scale_factor)   # skeleton y data
    lm_z = (all_trajectories[:, 0:self.tracked_landmark_count, z_axis] * flip_z * scale_factor)   # skeleton z data
    

    # convert landmark trajectory arrays to df and merge together
    x_df = pd.DataFrame(lm_x, columns = [name + "_x" for name in self.tracked_landmarks])
    y_df = pd.DataFrame(lm_y, columns = [name + "_y" for name in self.tracked_landmarks])
    z_df = pd.DataFrame(lm_z, columns = [name + "_z" for name in self.tracked_landmarks])
    merged_trajectories = pd.concat([x_df, y_df, z_df],axis = 1, join = "inner")    


    # add in Frame Number and Time stamp 
    merged_trajectories["Frame"] = [str(i) for i in range(0, len(merged_trajectories))]
    merged_trajectories["Time"] = merged_trajectories["Frame"].astype(float) / float(self.camera_rate)        

    # get the correct order for all dataframe columns
    column_order = []
    for marker in self.tracked_landmarks:
        column_order.append(marker + "_x")
        column_order.append(marker + "_y")
        column_order.append(marker + "_z")

    # Add Frame and Time
    column_order.insert(0, "Frame")
    column_order.insert(1, "Time")

    # reorder the dataframe, note frame number in 0 position remains
    merged_trajectories = merged_trajectories.reindex(columns=column_order)

    return merged_trajectories



def create_trajectory_trc(skeleton_data_frame, keypoints_names, frame_rate, data_array_folder_path):
    
    #Header
    data_rate = camera_rate = orig_data_rate = frame_rate
    num_frames = len(skeleton_data_frame)
    num_frame_range = range(num_frames)
    num_markers = len(keypoints_names)
    units = 'mm'
    orig_data_start_frame = num_frame_range[0]
    orig_num_frames = num_frame_range[-1]
    
    trc_filename = 'skel_trace.trc'
    trc_path = data_array_folder_path/trc_filename

    with open(trc_path, 'wt', newline='', encoding='utf-8') as out_file:
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

                # and finally actually write the trajectories
        for row in range(0, len(skeleton_data_frame)):
            tsv_writer.writerow(skeleton_data_frame.iloc[row].tolist())

        f = 2 


def flatten_mediapipe_data(skeleton_3d_data):
    num_frames = skeleton_3d_data.shape[0]
    num_markers = skeleton_3d_data.shape[1]

    skeleton_data_flat = skeleton_3d_data.reshape(num_frames,num_markers*3)

    return skeleton_data_flat


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

    skel_body_points = slice_mediapipe_data(skel3d_data,33)
    skel_3d_flat = flatten_mediapipe_data(skel_body_points)

#create_dataframe(skel_3d_flat,30)
    skel_3d_flat_dataframe = pd.DataFrame(skel_3d_flat)

    create_trajectory_trc(skel_3d_flat_dataframe,mediapipe_indices, 30, data_array_folder_path)


