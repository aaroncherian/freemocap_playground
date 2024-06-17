import numpy as np
import scipy.signal as signal
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

def compute_periodogram(data, fs):
    # Compute the Fourier transform of the data
    G = fft(data)
    # Compute the power spectral density
    psd = np.abs(G) ** 2
    # Get the frequencies corresponding to the PSD values
    freqs = fftfreq(len(data), 1/fs)
    return freqs[:len(freqs)//2], psd[:len(psd)//2]

def find_optimal_cutoff(freqs, psd, threshold=0.95):
    cumulative_power = np.cumsum(psd)
    total_power = cumulative_power[-1]
    # Define a threshold for cutoff frequency based on cumulative power
    optimal_cutoff_index = np.argmax(cumulative_power >= threshold * total_power)
    optimal_cutoff_frequency = freqs[optimal_cutoff_index]
    return optimal_cutoff_frequency

def butterworth_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

from pathlib import Path
from freemocap_utils.mediapipe_skeleton_builder import mediapipe_indices
# Example usage with synthetic data
path_to_freemocap_folder = Path(r"D:\2024-04-25_P01\1.0_recordings\sesh_2024-04-25_15_44_19_P01_WalkRun_Trial1")
path_to_freemocap_output_data = path_to_freemocap_folder/'output_data'/'raw_data'/'mediapipe3dData_numFrames_numTrackedPoints_spatialXYZ.npy'
freemocap_data = np.load(path_to_freemocap_output_data)
freemocap_data_ankle = freemocap_data[:, mediapipe_indices.index('left_heel'), 1]

y = freemocap_data_ankle
fs = 30

# Compute periodogram
freqs, psd = compute_periodogram(y, fs)

# Find optimal cutoff frequency with a threshold closer to typical noise reduction
optimal_cutoff = find_optimal_cutoff(freqs, psd, threshold=0.90)
print(f'Optimal Cutoff Frequency: {optimal_cutoff} Hz')

# Apply Butterworth filter with the optimal cutoff frequency
filtered_data = butterworth_filter(y, optimal_cutoff, fs)

# Plotting the results
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(y, label='Noisy data')
plt.plot(filtered_data, label='Filtered data')
plt.legend()
plt.title('Filtered Data with Optimal Cutoff Frequency')

plt.subplot(2, 1, 2)
plt.plot(freqs, psd)
plt.axvline(optimal_cutoff, color='r', linestyle='--', label=f'Optimal Cutoff: {optimal_cutoff:.2f} Hz')
plt.legend()
plt.title('Power Spectral Density and Optimal Cutoff Frequency')
plt.tight_layout()
plt.show()
