from pathlib import Path
import numpy as np
path_to_data = Path(r"D:\2025-04-23_atc_testing\freemocap\2025-04-23_19-11-05-612Z_atc_test_walk_trial_2\output_data\raw_data\mediapipe_3dData_numFrames_numTrackedPoints_spatialXYZ.npy")

data_2d = np.load(path_to_data)


num_nans = np.isnan(data_2d).sum()
print(f"Total number of NaNs: {num_nans}")