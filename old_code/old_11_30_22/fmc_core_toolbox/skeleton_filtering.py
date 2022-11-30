

from scipy import signal
import numpy as np
from rich.progress import track

def butter_lowpass_filter(data:np.ndarray, cutoff:float, sampling_rate:int, order:int):
    """ Run a low pass butterworth filter on a single column of data"""
    nyquist_freq = 0.5*sampling_rate
    normal_cutoff = cutoff / nyquist_freq
    # Get the filter coefficients 
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, data)
    return y


def filter_freemocap_data(freemocap_marker_data:np.ndarray, cutoff:float, sampling_rate:int, order:int) -> np.ndarray:
    """ Take in a 3d skeleton numpy array and run a low pass butterworth filter on each marker in the data"""
    num_frames = freemocap_marker_data.shape[0]
    num_markers = freemocap_marker_data.shape[1]
    filtered_data = np.empty((num_frames,num_markers,3))

    for marker in track(range(num_markers), description = 'Filtering Data'):
        for x in range(3):
            filtered_data[:,marker,x] = butter_lowpass_filter(freemocap_marker_data[:,marker,x],cutoff,sampling_rate,order)

    
    return filtered_data

if __name__ == '__main__':
        

    from pathlib import Path

    freemocap_data_folder_path = Path(r'D:\ValidationStudy2022\FreeMocap_Data')

        
    sessionID = 'sesh_2022-05-24_15_55_40_JSM_T1_BOS'
    data_array_folder = 'DataArrays'
    array_name = 'skel_3d_interpolated.npy'

    data_array_folder_path = freemocap_data_folder_path / sessionID / data_array_folder
    freemocap_marker_data = np.load(data_array_folder_path / array_name)

    
    sampling_rate = 30
    cutoff = 1
    order = 4

    filtered_data = filter_freemocap_data(freemocap_marker_data,cutoff,sampling_rate,order)
    f = 2

