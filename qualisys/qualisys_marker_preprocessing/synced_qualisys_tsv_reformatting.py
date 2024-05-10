import pandas as pd
import numpy as np
from pathlib import Path



def reformat_synced_qualisys_tsv_data_from_folder(path_to_recording_folder, tsv_name:str):

    path_to_qualisys_folder = path_to_recording_folder / 'qualisys_data'
    # path_to_save_numpy_array = path_to_qualisys_folder / 'qualisys_markers.npy'
    path_to_tsv = path_to_qualisys_folder / tsv_name

    original_qualisys_dataframe =  pd.read_csv(path_to_tsv, sep='\t')
    # Drop the 'Frame' and 'Time' columns
    original_qualisys_dataframe.drop(columns=['Frame', 'Time', 'unix_timestamps'] + [col for col in original_qualisys_dataframe.columns if 'Unnamed' in col], inplace=True)


    # Create the reorganized_data list with marker names as strings
    reorganized_qualisys_data= [
        [frame, col.split(' ')[0], row[col], row[f"{col.split(' ')[0]} Y"], row[f"{col.split(' ')[0]} Z"]]
        for frame, row in original_qualisys_dataframe.iterrows() for col in original_qualisys_dataframe.columns[::3]
    ]

    reorganized_qualisys_dataframe = pd.DataFrame(reorganized_qualisys_data, columns=['frame', 'marker', 'x', 'y', 'z'])
    reorganized_qualisys_dataframe.to_csv(path_to_qualisys_folder / 'qualisys_markers_dataframe.csv', index=False)


def reformat_synced_qualisys_data_as_csv(pasynced_qualisys_marker_dataframe):
    synced_qualisys_marker_dataframe.drop(columns=['Frame', 'Time', 'unix_timestamps'] + [col for col in synced_qualisys_marker_dataframe.columns if 'Unnamed' in col], inplace=True)
    # Create the reorganized_data list with marker names as strings
    reorganized_qualisys_data= [
        [frame, col.split(' ')[0], row[col], row[f"{col.split(' ')[0]} Y"], row[f"{col.split(' ')[0]} Z"]]
        for frame, row in synced_qualisys_marker_dataframe.iterrows() for col in synced_qualisys_marker_dataframe.columns[::3]
    ]

    reorganized_qualisys_dataframe = pd.DataFrame(reorganized_qualisys_data, columns=['frame', 'marker', 'x', 'y', 'z'])
    return reorganized_qualisys_dataframe



if __name__ == '__main__':
    # path_to_recording_folder = Path(r"D:\2023-06-07_TF01\1.0_recordings\treadmill_calib\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1")
    # tsv_name = 'flexion_neutral_trial_1_tracked_with_header_synchronized.tsv'
    path_to_recording_folder = Path(r'D:\2023-06-07_TF01\1.0_recordings\treadmill_calib\sesh_2023-06-07_12_38_16_TF01_leg_length_neg_5_trial_1')
    tsv_name = 'synchronized_markers.tsv'
    reformat_synced_qualisys_tsv_data(path_to_recording_folder, tsv_name)



# Convert the reorganized_data list to a NumPy array
# reorganized_array = np.array(reorganized_data_array)

# # Reshape the array to [frame, marker, dimension]
# reshaped_array = reorganized_array.reshape(df.shape[0], len(unique_markers), 5)[:, :, 2:]

# np.save(path_to_save_numpy_array, reshaped_array)
# f = 2
