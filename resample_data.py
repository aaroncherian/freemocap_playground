from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# from freemocap_utils import freemocap_data_loader
# from freemocap_utils.mediapipe_skeleton_builder import mediapipe_indices
from scipy import signal
import scipy
from freemocap_utils.skeleton_interpolation import interpolate_freemocap_data


def butter_lowpass_filter(data, cutoff, sampling_rate, order):
    """ Run a low pass butterworth filter on a single column of data"""
    nyquist_freq = 0.5*sampling_rate
    normal_cutoff = cutoff / nyquist_freq
    # Get the filter coefficients 
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, data)
    return y


def butterworth_filter_data(data_to_filter, cutoff, sampling_rate, order):
    """ Take in a 3d skeleton numpy array and run a low pass butterworth filter on each marker in the data"""
    num_frames = data_to_filter.shape[0]
    num_markers = data_to_filter.shape[1]
    filtered_data = np.empty((num_frames,num_markers,3))

    for marker in range(num_markers):
        for x in range(3):
            filtered_data[:,marker,x] = butter_lowpass_filter(data_to_filter[:,marker,x],cutoff,sampling_rate,order)
    
    return filtered_data
    

def resample_data(data_to_resample:np.ndarray, original_framerate:float, framerate_to_resample_to:float):
    num_samples = data_to_resample.shape[0]
    num_markers = data_to_resample.shape[1]
    num_dimensions = data_to_resample.shape[2]

    #create time vectors that go from 0 to the length in seconds of the recording
    original_end_time_point = num_samples/original_framerate #calculation to get the length of the original recording
    original_time_array = np.arange(0,original_end_time_point,1/original_framerate) #a time array for the original data with the original framerate
    resampled_time_array = np.arange(0,original_time_array[-1],1/framerate_to_resample_to) #a time vector that the original data will be matched to with a new framerate for resampling
        #resampled_time_array is bounded by the last point of the original time array - this prevents the resampled time array from going past those original bounds, which was throwing an error in interpolate

    resampled_data = np.empty([resampled_time_array.shape[0],num_markers,num_dimensions])

    for marker in range (num_markers):
        for dimension in range(num_dimensions):
            interpolation_function = scipy.interpolate.interp1d(original_time_array,data_to_resample[:,marker,dimension])
            resampled_data[:,marker,dimension] = interpolation_function(resampled_time_array)

    return resampled_data



path_to_freemocap_session_folder = Path(r'D:\2024-04-25_P01\1.0_recordings\P01_WalkRun_Trial1_four_cameras')
freemocap_data = np.load(path_to_freemocap_session_folder/'output_data'/'mediaPipeSkel_3d_body_hands_face.npy')

path_to_qualisys_session_folder = path_to_freemocap_session_folder
qualisys_data = np.load(path_to_qualisys_session_folder/'qualisys_data'/'qualisys_joint_centers_3d_xyz.npy')
interpolated_qualisys_data = interpolate_freemocap_data(qualisys_data)
filtered_qualisys_data = butterworth_filter_data(interpolated_qualisys_data,cutoff=6, sampling_rate=29.954299049366554, order=4)

freemocap_framerate = 29.954282127686497
qualisys_framerate = 29.954299049366554

resampled_qualisys_data = resample_data(data_to_resample=filtered_qualisys_data, original_framerate=qualisys_framerate, framerate_to_resample_to=freemocap_framerate)
np.save(path_to_qualisys_session_folder/'qualisys_data'/'downsampled_qualisys_joint_centers_3d_xyz.npy', resampled_qualisys_data)

