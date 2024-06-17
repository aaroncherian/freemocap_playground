from skellyforge.freemocap_utils.postprocessing_widgets.postprocessing_functions.filter_data import butter_lowpass_filter, filter_skeleton_data

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# # Plot frequency spectrum of the original and filtered signals
# def plot_frequency_spectrum(data, sampling_rate, title="Frequency Spectrum"):
#     freqs, psd = signal.welch(data, sampling_rate)
#     plt.figure()
#     plt.semilogy(freqs, psd)
#     plt.title(title)
#     plt.xlabel('Frequency (Hz)')
#     plt.ylabel('Power Spectral Density')
#     plt.show()

# def plot_frequency_spectrums(data1, data2, sampling_rate, title1="Frequency Spectrum 1", title2="Frequency Spectrum 2"):
#     freqs1, psd1 = signal.welch(data1, sampling_rate)
#     freqs2, psd2 = signal.welch(data2, sampling_rate)
    
#     plt.figure(figsize=(12, 6))
    
#     x_range = max(max(freqs1), max(freqs2))

#     plt.subplot(2, 1, 1)
#     plt.semilogy(freqs1, psd1)
#     plt.title(title1)
#     plt.xlabel('Frequency (Hz)')
#     plt.ylabel('Power Spectral Density')
#     plt.xlim([0, x_range])  # Set x-axis range
    
#     plt.subplot(2, 1, 2)
#     plt.semilogy(freqs2, psd2)
#     plt.title(title2)
#     plt.xlabel('Frequency (Hz)')
#     plt.ylabel('Power Spectral Density')
#     plt.xlim([0, x_range])  # Set x-axis range
    
#     plt.tight_layout()
#     plt.show()
    
# # # Test the butter_lowpass_filter function with known frequencies
# # sampling_rate = 30  # 100 Hz sampling rate
# # cutoff = 7  # 7 Hz cutoff frequency
# # order = 4  # 4th order Butterworth filter

# # # Create a synthetic signal with known frequencies (e.g., 5 Hz and 15 Hz)
# # t = np.linspace(0, 1.0, sampling_rate, endpoint=False)
# # synthetic_signal = np.sin(2 * np.pi * 2 * t) + 0.5 * np.sin(2 * np.pi * 14 * t)

# # # Apply the low-pass filter to the synthetic signal
# # filtered_signal = butter_lowpass_filter(synthetic_signal, cutoff, sampling_rate, order)

# # # Plot the original and filtered signals
# # plt.figure(figsize=(12, 6))

# # plt.subplot(2, 1, 1)
# # plt.plot(t, synthetic_signal, label='Original Signal')
# # plt.title('Original Signal (5 Hz and 15 Hz)')
# # plt.xlabel('Time [s]')
# # plt.ylabel('Amplitude')
# # plt.legend()

# # plt.subplot(2, 1, 2)
# # plt.plot(t, filtered_signal, label='Filtered Signal', color='red')
# # plt.title('Filtered Signal (7 Hz cutoff)')
# # plt.xlabel('Time [s]')
# # plt.ylabel('Amplitude')
# # plt.legend()

# # plt.tight_layout()
# # plt.show()



# # plot_frequency_spectrum(synthetic_signal, sampling_rate, title="Original Signal Frequency Spectrum")
# # plot_frequency_spectrum(filtered_signal, sampling_rate, title="Filtered Signal Frequency Spectrum")

# from pathlib import Path
# from freemocap_utils.mediapipe_skeleton_builder import mediapipe_indices

# path_to_freemocap_folder = Path(r"D:\2024-04-25_P01\1.0_recordings\sesh_2024-04-25_15_44_19_P01_WalkRun_Trial1")
# # path_to_freemocap_output_data = path_to_freemocap_folder/'output_data'/'mediapipe_body_3d_xyz.npy'
# path_to_freemocap_output_data = path_to_freemocap_folder/'output_data'/'raw_data'/'mediapipe3dData_numFrames_numTrackedPoints_spatialXYZ.npy'

# freemocap_data = np.load(path_to_freemocap_output_data)

# freemocap_data_ankle = freemocap_data[:, mediapipe_indices.index('left_ankle'), 1]
# filtered_data_ankle = butter_lowpass_filter(freemocap_data_ankle, 3.5, 30, 4)



# # plot_frequency_spectrums(freemocap_data_ankle, filtered_data_ankle, 30, title1="Original Signal Frequency Spectrum", title2="Filtered Signal Frequency Spectrum")


# filtered_skeleton_data = filter_skeleton_data(freemocap_data, order=4, cutoff=3.5, sampling_rate=30)

# np.save(path_to_freemocap_folder/'output_data'/'butterworth_test.npy', filtered_skeleton_data[:,0:33,:])

# # plt.figure(figsize=(12, 6))

# # plt.subplot(2, 1, 1)
# # plt.plot(freemocap_data_ankle, label='Original Signal')
# # plt.title('Original Signal (5 Hz and 15 Hz)')
# # plt.xlabel('Time [s]')
# # plt.ylabel('Amplitude')
# # plt.legend()

# # plt.subplot(2, 1, 2)
# # plt.plot(filtered_data_ankle, label='Filtered Signal', color='red')
# # plt.title('Filtered Signal (7 Hz cutoff)')
# # plt.xlabel('Time [s]')
# # plt.ylabel('Amplitude')
# # plt.legend()

# # plt.tight_layout()
# # plt.show()


# # plot_frequency_spectrum(freemocap_data_ankle, 30, title="Original Signal Frequency Spectrum")
# # plot_frequency_spectrum(filtered_data_ankle, 30, title="Filtered Signal Frequency Spectrum")

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import linregress
from pathlib import Path
from freemocap_utils.mediapipe_skeleton_builder import mediapipe_indices

# Butterworth filter function
def butter_lowpass_filter(data, cutoff, sampling_rate, order):
    nyquist_freq = 0.5 * sampling_rate
    normal_cutoff = cutoff / nyquist_freq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, data)
    return y

# Compute RMSE of second derivatives (accelerations)
def compute_rmse(original_data, filtered_data):
    return np.sqrt(np.mean((original_data - filtered_data) ** 2))

def compute_second_derivative(data, sampling_rate):
    time_step = 1 / sampling_rate
    second_derivative = np.gradient(np.gradient(data, time_step), time_step)
    return second_derivative

# Implement Bing Yu's method
def find_optimal_cutoff(data, sampling_rate, order, cutoff_range):
    rmses = []
    filtered_signals = []
    
    # Compute the second derivative of the raw data
    original_second_derivative = compute_second_derivative(data, sampling_rate)
    
    for cutoff in cutoff_range:
        filtered_data = butter_lowpass_filter(data, cutoff, sampling_rate, order)
        filtered_second_derivative = compute_second_derivative(filtered_data, sampling_rate)
        rmse = compute_rmse(original_second_derivative, filtered_second_derivative)
        rmses.append(rmse)
        filtered_signals.append(filtered_data)
    
    rmses = np.array(rmses)
    cutoff_range = np.array(cutoff_range)

    # Plot RMSE vs. cutoff frequency
    plt.plot(cutoff_range, rmses, 'b', label='RMSE of Second Derivative')
    plt.xlabel('Cutoff Frequency (Hz)')
    plt.ylabel('RMSE')
    plt.title('RMSE vs. Cutoff Frequency')
    
    # Find the cutoff frequency that minimizes the RMSE
    optimal_cutoff_idx = np.argmin(rmses)
    f_ideal = cutoff_range[optimal_cutoff_idx]
    
    plt.axvline(x=f_ideal, color='r', linestyle='--', label=f'Optimal Cutoff Frequency: {f_ideal} Hz')
    
    plt.legend()
    plt.show()
    
    return f_ideal, filtered_signals[optimal_cutoff_idx]

# Load FreeMoCap data
path_to_freemocap_folder = Path(r"D:\2024-04-25_P01\1.0_recordings\sesh_2024-04-25_15_44_19_P01_WalkRun_Trial1")
path_to_freemocap_output_data = path_to_freemocap_folder/'output_data'/'raw_data'/'mediapipe3dData_numFrames_numTrackedPoints_spatialXYZ.npy'
freemocap_data = np.load(path_to_freemocap_output_data)
freemocap_data_ankle = freemocap_data[:, mediapipe_indices.index('left_ankle'), 1]

# Define parameters for the example
sampling_rate = 30  # Set sampling rate
order = 4  # Butterworth filter order
cutoff_range = np.arange(1, 10.1, 0.1)  # Cutoff frequency range from 1 to 10 Hz with 0.1 Hz intervals

# Find the optimal cutoff frequency
optimal_cutoff, optimal_filtered_data = find_optimal_cutoff(freemocap_data_ankle, sampling_rate, order, cutoff_range)

print(f"Optimal Cutoff Frequency: {optimal_cutoff} Hz")

# Plot the original and filtered signal
plt.figure(figsize=(10, 6))
plt.plot(freemocap_data_ankle, label='Original Signal')
plt.plot(optimal_filtered_data, label=f'Filtered Signal (Cutoff: {optimal_cutoff} Hz)')
plt.legend()
plt.xlabel('Time (frames)')
plt.ylabel('Amplitude')
plt.title('Original vs. Filtered Signal')
plt.show()
b