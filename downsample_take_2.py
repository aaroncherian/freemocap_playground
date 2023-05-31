from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from freemocap_utils import freemocap_data_loader
from freemocap_utils.mediapipe_skeleton_builder import mediapipe_indices
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


def filter_skeleton(skeleton_3d_data, cutoff, sampling_rate, order):
    """ Take in a 3d skeleton numpy array and run a low pass butterworth filter on each marker in the data"""
    num_frames = skeleton_3d_data.shape[0]
    num_markers = skeleton_3d_data.shape[1]
    filtered_data = np.empty((num_frames,num_markers,3))

    for marker in range(num_markers):
        for x in range(3):
            filtered_data[:,marker,x] = butter_lowpass_filter(skeleton_3d_data[:,marker,x],cutoff,sampling_rate,order)
    
    return filtered_data

def downsample_data(data,time_old,time_new):
    num_markers = data.shape[1]
    num_dimensions = data.shape[2]

    downsampled_data = np.empty([time_new.shape[0],num_markers,num_dimensions])
    for marker in range (num_markers):
        for dimension in range(num_dimensions):

            interp_function = scipy.interpolate.interp1d(time_old,data[:,marker,dimension], fill_value = 'extrapolate')
            downsampled_data[:,marker,dimension] = interp_function(time_new)
    return downsampled_data


path_to_freemocap_session_folder = Path(r'"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_14_40_56_MDN_NIH_Trial2"')
freemocap_data = np.load(path_to_freemocap_session_folder/'output_data'/'mediaPipeSkel_3d_body_hands_face.npy')

num_markers = freemocap_data.shape[1]
num_dimensions = freemocap_data.shape[2]

freemocap_framerate = 29.9778768021153


time_old = np.arange(0,freemocap_data.shape[0]/freemocap_framerate,1/freemocap_framerate) #use shape/framerate to get the exact number of frames as in the data
time_new = np.arange(0,freemocap_data.shape[0]/freemocap_framerate,1/30)


downsampled_freemocap_data = downsample_data(freemocap_data,time_old, time_new)

path_to_qualisys_session_folder = Path(r"D:\ValidationStudy2022\FreeMocap_Data\qualisys_sesh_2022-05-24_16_02_53_JSM_T1_WalkRun")
qualisys_data = np.load(path_to_qualisys_session_folder/'DataArrays'/'qualisys_origin_aligned_skeleton_3D.npy')
interpolated_qualisys_data = interpolate_freemocap_data(qualisys_data)
filtered_qualisys_skeleton = filter_skeleton(interpolated_qualisys_data,6,300,4)

time_old_q = np.arange(0,182.13,1/300)
time_new_q = np.arange(0,182.13,1/30)

downsampled_qualisys_data = downsample_data(filtered_qualisys_skeleton,time_old_q,time_new_q)

figure = plt.figure()


trajectories_x_ax = figure.add_subplot(311)
trajectories_y_ax = figure.add_subplot(312)
trajectories_z_ax = figure.add_subplot(313)

trajectories_x_ax.set_ylabel('X Axis (mm)')
trajectories_y_ax.set_ylabel('Y Axis (mm)')
trajectories_z_ax.set_ylabel('Z Axis (mm)')

ax_list = [trajectories_x_ax,trajectories_y_ax,trajectories_z_ax]

num_to_plot = 1180

for dimension,ax in enumerate(ax_list):
    # ax.plot(time_old,freemocap_data[:,11,dimension], color = 'blue', alpha = 1, label = 'original')
    # #ax.plot(t_old,interpolated_qualisys_data[:,5,dimension], color = 'red', alpha = 1, label = 'interpolated')
    # #ax.plot(t_old,filtered_qualisys_skeleton[:,5,dimension],color = 'red', alpha = .8, label = 'filtered')
    # ax.plot(time_new,downsampled_freemocap_data[:,11,dimension], color = 'red', label = 'downsampled')
    ax.plot(downsampled_freemocap_data[num_to_plot:,11,0]- downsampled_freemocap_data[num_to_plot,11,0])
    ax.plot(downsampled_qualisys_data[:,4,0] - downsampled_qualisys_data[0,4,0])

    ax.legend()


np.save(path_to_qualisys_session_folder/'DataArrays'/'test_qual_ds.npy', downsampled_qualisys_data)
np.save(path_to_freemocap_session_folder/'DataArrays'/'test_fmc_ds.npy', downsampled_freemocap_data)


plt.show()