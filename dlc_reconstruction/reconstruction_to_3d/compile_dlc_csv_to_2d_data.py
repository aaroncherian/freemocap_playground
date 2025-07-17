import pandas as pd
import numpy as np
from pathlib import Path

def compile_dlc_csvs(path_to_folder_with_dlc_csvs:Path,
                     confidence_threshold:float = 0.5,
                     ):


    # Filtered csv list
    csv_list = sorted(list(path_to_folder_with_dlc_csvs.glob('*.csv')))

    # Initialize an empty list to hold dataframes
    dfs = []

    for csv in csv_list:
        # Read each csv into a dataframe with a multi-index header
        df = pd.read_csv(csv, header=[1, 2])
        
        # Drop the first column (which just has the headers )
        df = df.iloc[:, 1:]
        
        # Check if data shape is as expected
        if df.shape[1] % 3 != 0:
            print(f"Unexpected number of columns in {csv}: {df.shape[1]}")
            continue
        
        try:
            # Convert the df into a 4D numpy array of shape (1, num_frames, num_markers, 3) and append to dfs
            dfs.append(df.values.reshape(1, df.shape[0], df.shape[1]//3, 3))
        except ValueError as e:
            print(f"Reshape failed for {csv} with shape {df.shape}: {e}")


    # Concatenate all the arrays along the first axis (camera axis)
    dlc_2d_array_with_confidence = np.concatenate(dfs, axis=0)

    confidence_thresholded_dlc_2d_array_XYC = apply_confidence_threshold(array=dlc_2d_array_with_confidence, threshold=confidence_threshold)
    # final_thresholded_array = apply_confidence_threshold(final_array, 0.6)

    confidence_thresholded_dlc_2d_array_XY = confidence_thresholded_dlc_2d_array_XYC[:,:,:,:2]

    return confidence_thresholded_dlc_2d_array_XY


import numpy as np

def apply_confidence_threshold(array, threshold):
    """
    Set X,Y values to NaN where the corresponding confidence value is below threshold.
    """
    mask = array[..., 2] < threshold  # Shape: (num_cams, num_frames, num_markers)
    array[mask, 0] = np.nan  # Set X to NaN where confidence is low
    array[mask, 1] = np.nan  # Set Y to NaN where confidence is low
    return array



if __name__ == '__main__':

    path_to_folder_with_dlc_csvs = Path(r'D:\sfn\michael_wobble\recording_12_07_09_gmt-5__MDN_wobble_3\dlc_data')
    dlc_2d_array = compile_dlc_csvs(path_to_folder_with_dlc_csvs,
                                    confidence_threshold=.5)


    f = 2