import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import linregress
from scipy.signal import find_peaks
from pathlib import Path
from freemocap_utils.mediapipe_skeleton_builder import mediapipe_indices

# Load FreeMoCap data
path_to_freemocap_folder = Path(r"D:\2024-04-25_P01\1.0_recordings\sesh_2024-04-25_15_44_19_P01_WalkRun_Trial1")
path_to_freemocap_output_data = path_to_freemocap_folder/'output_data'/'raw_data'/'mediapipe3dData_numFrames_numTrackedPoints_spatialXYZ.npy'
freemocap_data = np.load(path_to_freemocap_output_data)
freemocap_data_ankle = freemocap_data[:, mediapipe_indices.index('left_heel'), 1]

# Butterworth filter function
def butter_lowpass_filter(data, cutoff, sampling_rate, order):
    nyquist_freq = 0.5 * sampling_rate
    normal_cutoff = cutoff / nyquist_freq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, data)
    return y

# Compute RMSE function
def compute_rmse(original_data, filtered_data):
    return np.sqrt(np.mean((original_data - filtered_data) ** 2))

# Find the first significant change in the slope (flattening point)
def find_flattening_point(rmses):
    differences = np.diff(rmses)
    second_derivative = np.diff(differences)
    peaks, _ = find_peaks(second_derivative)  # Negative to find valleys as peaks
    if len(peaks) > 0:
        return peaks[2]
    return None

# Improved method to identify the flattening point and fit regression line correctly
def find_optimal_cutoff(data, sampling_rate, order, cutoff_range):
    rmses = []
    filtered_signals = []
    
    for cutoff in cutoff_range:
        filtered_data = butter_lowpass_filter(data, cutoff, sampling_rate, order)
        rmse = compute_rmse(data, filtered_data)
        rmses.append(rmse)
        filtered_signals.append(filtered_data)
    
    rmses = np.array(rmses)
    cutoff_range = np.array(cutoff_range)

    # Plot RMSE vs. cutoff frequency
    plt.plot(cutoff_range, rmses, 'b', label='RMSE')
    plt.xlabel('Cutoff Frequency (Hz)')
    plt.ylabel('RMSE')
    plt.title('RMSE vs. Cutoff Frequency')
    
    # Identify the flattening point dynamically
    flattening_idx = find_flattening_point(rmses)
    if flattening_idx is None:
        raise ValueError("No flattening point found in the RMSE plot.")
    
    # Perform linear regression on the tail end of the RMSE plot from the flattening point onward
    tail_cutoff_range = cutoff_range[flattening_idx:]
    tail_rmses = rmses[flattening_idx:]
    slope, intercept, _, _, _ = linregress(tail_cutoff_range, tail_rmses)
    
    # Plot the regression line
    regression_line = slope * tail_cutoff_range + intercept
    plt.plot(tail_cutoff_range, regression_line, 'gray', linestyle='--', label='Regression Line')
    
    # Plot the threshold line
    plt.axhline(y=intercept, color='r', linestyle='--', label='Threshold')
    
    # Mark the tail end point
    plt.plot(cutoff_range[flattening_idx], rmses[flattening_idx], 'go', label='Tail End Start')
    
    plt.legend()
    plt.show()
    
    # Find the first cutoff frequency where RMSE exceeds the threshold
    threshold = intercept
    optimal_cutoff_idx = np.where(rmses > threshold)[0][-1]
    f_ideal = cutoff_range[optimal_cutoff_idx]
    
    return f_ideal, filtered_signals[optimal_cutoff_idx]

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
