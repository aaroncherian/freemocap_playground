import numpy as np 
from pathlib import Path

## MEDIAPIPE SLICING DATA
mediapipe_body_slice = slice(0,33)
mediapipe_hands_slice = slice(33,75)

## RECORDING DATA
path_to_recording = Path(r'C:\Users\aaron\freemocap_data\recording_sessions\freemocap_test_data_v1_4_6')

path_to_raw_data = path_to_recording/'output_data'/'raw_data'/'mediapipe_3dData_numFrames_numTrackedPoints_spatialXYZ.npy'
path_to_processed_data = path_to_recording/'output_data'/'mediapipe_skeleton_3d.npy'


## LOAD DATA

def load_and_slice_data(path_to_data:Path, body_slice:slice, hands_slice: slice):
    data = np.load(path_to_data)
    body_data = data[:, body_slice, :]
    hands_data = data[:, hands_slice, :]

    return data, body_data, hands_data

def calculate_jerk(position:np.ndarray): #NOTE: may want to divide by dt, but if we're just using this for comparison it may not be necessary. However if we do want dt, will either need timestamps or framerate as well
    velocity = np.diff(position, axis=0)
    acceleration = np.diff(velocity, axis=0)
    jerk = np.diff(acceleration, axis=0)

    return jerk
# raw_data, raw_body_data, raw_hands_data = load_and_slice_data(path_to_data=path_to_raw_data,
#                                                                                 body_slice=mediapipe_body_slice,
#                                                                                 hands_slice=mediapipe_hands_slice)


processed_data, processed_body_data, processed_hands_data = load_and_slice_data(path_to_data=path_to_processed_data,
                                                                                body_slice=mediapipe_body_slice,
                                                                                hands_slice=mediapipe_hands_slice)



processed_body_jerk = calculate_jerk(processed_body_data)
total_avg_jerk = np.mean(np.abs(processed_body_jerk))
avg_jerk_per_joint = np.mean(np.abs(processed_body_jerk), axis=(0, 2))  # Shape: (num_joints,)

# Print results
print(f"Total Average BODY Jerk: {total_avg_jerk}")
print(f"Average Jerk Per BODY Joint:\n {avg_jerk_per_joint}")

processed_hands_jerk = calculate_jerk(processed_hands_data)
total_avg_hands_jerk = np.mean(np.abs(processed_hands_jerk))

print(f"Total Average HANDS Jerk: {total_avg_hands_jerk}")

f =2 
