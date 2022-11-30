

import socket
from pathlib import Path

from scipy import signal
import numpy as np

def butter_lowpass_filter(data, cutoff, sampling_rate, order):
    """ Run a low pass butterworth filter on a single column of data"""
    nyquist_freq = 0.5*sampling_rate
    normal_cutoff = cutoff / nyquist_freq
    # Get the filter coefficients 
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, data)
    return y


def filter_skeleton(skeleton_3d_data, cutoff, sampling_rate, order):
    """ Take in a 3d skeleton numpy array and run a low pass butterworth filter on each marker in the data"""
    num_frames = skeleton_3d_data.shape[0]
    num_markers = skeleton_3d_data.shape[1]
    filtered_data = np.empty((num_frames,num_markers,3))

    for marker in range(num_markers):
        for x in range(3):
            filtered_data[:,marker,x] = butter_lowpass_filter(skeleton_3d_data[:,marker,x],cutoff,sampling_rate,order)

    
    return filtered_data

if __name__ == '__main__':

    this_computer_name = socket.gethostname()

    if this_computer_name == 'DESKTOP-F5LCT4Q':
        #freemocap_validation_data_path = Path(r"C:\Users\aaron\Documents\HumonLab\Spring2022\ValidationStudy\FreeMocap_Data")
        #freemocap_data_folder_path = Path(r'D:\freemocap2022\FreeMocap_Data')
        freemocap_data_folder_path = Path(r'D:\ValidationStudy2022\FreeMocap_Data')
    else:
        freemocap_data_folder_path = Path(r'C:\Users\Aaron\Documents\sessions\FreeMocap_Data')
        
    sessionID = 'sesh_2022-05-24_15_55_40_JSM_T1_BOS'
    data_array_folder = 'DataArrays'
    array_name = 'skel_3d_interpolated.npy'

    data_array_folder_path = freemocap_data_folder_path / sessionID / data_array_folder
    skel3d_data = np.load(data_array_folder_path / array_name)

    
    sampling_rate = 30
    cutoff = 1
    order = 4

    filtered_data = filter_skeleton(skel3d_data,cutoff,sampling_rate,order)
    f = 2

