import numpy as np 
from pathlib import Path

## MEDIAPIPE SLICING DATA
mediapipe_body_slice = slice(0,33)
mediapipe_hands_slice = slice(33,75)

## RECORDING DATA
path_to_recording = Path(r'C:\Users\aaron\freemocap_data\recording_sessions\freemocap_test_data_v1_4_6')

path_to_processed_data = path_to_recording/'output_data'/'mediapipe_skeleton_3d.npy'
path_to_rigid_data = path_to_recording/'output_data'/'mediapipe_rigid_bones_3d.npy'


## LOAD DATA

def load_and_slice_data(path_to_data:Path, body_slice:slice, hands_slice: slice):
    data = np.load(path_to_data)
    body_data = data[:, body_slice, :]
    hands_data = data[:, hands_slice, :]

    return data, body_data, hands_data

processed_data, processed_body_data, processed_hands_data = load_and_slice_data(path_to_data=path_to_processed_data,
                                                                                body_slice=mediapipe_body_slice,
                                                                                hands_slice=mediapipe_hands_slice)

rigid_body_data = np.load(path_to_rigid_data)

