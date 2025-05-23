import pandas as pd
import numpy as np
from pathlib import Path



def reformat_qualisys_tsv_data(path_to_recording_folder, tsv_name:str):

    path_to_qualisys_folder = path_to_recording_folder / 'qualisys_data'
    path_to_save_numpy_array = path_to_qualisys_folder / 'qualisys_markers.npy'
    path_to_tsv = path_to_qualisys_folder / tsv_name

    original_qualisys_dataframe =  pd.read_csv(path_to_tsv, sep='\t')
    # Drop the 'Frame' and 'Time' columns
    original_qualisys_dataframe.drop(columns=['Frame', 'Time'], inplace=True)

    # Create the reorganized_data list with marker names as strings
    reorganized_qualisys_data= [
        [frame, col.split(' ')[0], row[col], row[f"{col.split(' ')[0]} Y"], row[f"{col.split(' ')[0]} Z"]]
        for frame, row in original_qualisys_dataframe.iterrows() for col in original_qualisys_dataframe.columns[::3]
    ]

    reorganized_qualisys_dataframe = pd.DataFrame(reorganized_qualisys_data, columns=['frame', 'marker', 'x', 'y', 'z'])
    reorganized_qualisys_dataframe.to_csv(path_to_qualisys_folder / 'qualisys_markers_dataframe.csv', index=False)

if __name__ == '__main__':
    path_to_recording_folder = Path(r"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_13_48_44_MDN_treadmill_2")
    tsv_name = 'MDN_treadmill_2_tracked_2.tsv'
    reformat_qualisys_tsv_data(path_to_recording_folder, tsv_name)



# Convert the reorganized_data list to a NumPy array
# reorganized_array = np.array(reorganized_data_array)

# # Reshape the array to [frame, marker, dimension]
# reshaped_array = reorganized_array.reshape(df.shape[0], len(unique_markers), 5)[:, :, 2:]

# np.save(path_to_save_numpy_array, reshaped_array)
# f = 2
