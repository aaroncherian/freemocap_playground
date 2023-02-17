import numpy as np 
from scipy.io import loadmat

from pathlib import Path

path_to_session_folder = Path(r'D:\ValidationStudy2022\FreeMocap_Data\qualisys_sesh_2022-05-24_16_02_53_JSM_T1_NIH')
path_to_mat_file = path_to_session_folder/'DataArrays'/'qualisys_markers_3d.mat'

qualisys_data_from_mat = loadmat(path_to_mat_file)

qualisys_data = qualisys_data_from_mat['mat_data_reshaped']

np.save(path_to_session_folder/'DataArrays'/'qualisys_markers_3d.npy', qualisys_data)



f = 2
