from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scipy import signal
import scipy
from freemocap_utils.skeleton_interpolation import interpolate_freemocap_data
from ast import literal_eval

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


def get_freemocap_framerate(path_to_freemocap_recording_folder):
    recording_id = path_to_freemocap_recording_folder.stem
    path_to_framerate_file = path_to_freemocap_recording_folder / f"{recording_id}_framerate.txt"
    
    with open(path_to_framerate_file, 'r') as f:
        framerate_str = f.read()
        framerate_dict = literal_eval(framerate_str)  # convert string to dictionary
    
    framerate = framerate_dict['framerate']
    return framerate



path_to_freemocap_recording_folder = Path(r'D:\2023-06-07_TF01\1.0_recordings\treadmill_calib\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1')


freemocap_framerate = get_freemocap_framerate(path_to_freemocap_recording_folder)


path_to_qualisys_data = path_to_freemocap_recording_folder / 'qualisys' / 'qualisys_joint_centers_3d_xyz.npy'
path_to_save_data = path_to_freemocap_recording_folder / 'qualisys' / 'resampled_qualisys_joint_centers_3d_xyz.npy'

qualisys_joint_center_data = np.load(path_to_qualisys_data)

qualisys_framerate = 300

interpolated_qualisys_data = interpolate_freemocap_data(qualisys_joint_center_data)
filtered_qualisys_data = butterworth_filter_data(interpolated_qualisys_data,cutoff=6, sampling_rate=qualisys_framerate, order=4)
resampled_qualisys_data = resample_data(data_to_resample=filtered_qualisys_data, original_framerate=qualisys_framerate, framerate_to_resample_to=freemocap_framerate)


f = 2
np.save(path_to_save_data, resampled_qualisys_data)
# np.save(path_to_qualisys_session_folder/'output_data'/'downsampled_qualisys_skel_3d.npy', resampled_qualisys_data)

