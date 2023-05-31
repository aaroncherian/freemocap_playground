import numpy as np
import scipy.io as sio

from pathlib import Path

freemocap_data_folder_path = Path(r'D:\2023-05-17_MDN_NIH_data\qtm_data')

session_ID = 'qualisys_MDN_NIH_Trial2'
debug = True
#frame_to_use = 2000
freemocap_data_array_path = freemocap_data_folder_path/session_ID/'output_data'

#qualisys_data = np.load(freemocap_data_array_path/'qualisys_markers_3d.npy')
qualisys_data_path = freemocap_data_array_path/'qualisys_markers_3d.mat'
qualisys_mat_file = sio.loadmat(qualisys_data_path)
qualisys_data = qualisys_mat_file['mat_data_reshaped']
f = 2

qualisys_skel_save_path = freemocap_data_array_path/'qualisys_markers_3d.npy'

# np.save(qualisys_skel_save_path,qualisys_data)