import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import json
import numpy as np
import seaborn as sns

def aggregate_path_lengths(analysis_folders):
    # Lists to store path lengths
    freemocap_path_lengths = []
    qualisys_path_lengths = []

    # Process each analysis folder
    for path_to_analysis_folder in analysis_folders:
        # Check if path exists
        if not Path(path_to_analysis_folder).exists():
            print(f"Path {path_to_analysis_folder} does not exist.")
            continue
        
        # Load JSON file
        json_name = 'condition_data.json'
        json_path = path_to_analysis_folder / json_name
        
        # Check if JSON file exists
        if not json_path.exists():
            print(f"JSON file not found in {path_to_analysis_folder}.")
            continue

        # Read the JSON data
        with open(json_path, 'r') as file:
            json_data = json.load(file)
        
        # Extract and store path lengths (assuming JSON structure is known)
        if 'sesh' in str(path_to_analysis_folder).lower(): #this is a freemocap folder
            freemocap_path_lengths.append(pd.DataFrame(json_data['Path Lengths:'], index=[0]))
        elif 'qualisys' in str(path_to_analysis_folder).lower():
            qualisys_path_lengths.append(pd.DataFrame(json_data['Path Lengths:'], index=[0]))

    # Concatenate DataFrames
    freemocap_path_lengths = pd.concat(freemocap_path_lengths)
    qualisys_path_lengths = pd.concat(qualisys_path_lengths)

    return freemocap_path_lengths, qualisys_path_lengths

# Example usage
analysis_folders = [
    Path(r'D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_14_40_56_MDN_NIH_Trial2\data_analysis\analysis_2023-06-01_10_03_59'),
    Path(r'D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_14_53_48_MDN_NIH_Trial3\data_analysis\analysis_2023-06-01_10_12_24'),
    Path(r'D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_15_03_20_MDN_NIH_Trial4\data_analysis\analysis_2023-06-01_10_17_22'),
    Path(r'D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\qualisys_MDN_NIH_Trial2\data_analysis\analysis_2023-06-01_16_11_00'),
    Path(r'D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\qualisys_MDN_NIH_Trial3\data_analysis\analysis_2023-06-01_17_14_40'),
    Path(r'D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\qualisys_MDN_NIH_Trial4\data_analysis\analysis_2023-06-01_18_06_59')
    # ... more paths
]

freemocap_path_lengths, qualisys_path_lengths = aggregate_path_lengths(analysis_folders)

sns.set_style('whitegrid')

# Calculate mean and standard deviation
freemocap_mean = freemocap_path_lengths.mean()
freemocap_std = freemocap_path_lengths.std()

qualisys_mean = qualisys_path_lengths.mean()
qualisys_std = qualisys_path_lengths.std()

# Conditions
conditions = freemocap_mean.index
conditions = [condition.replace('/', '\n') for condition in conditions]

# Create figure with subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 10), sharey=True)

# Plotting Freemocap data
for index, row in freemocap_path_lengths.iterrows():
    axs[0].plot(conditions, row, '-o', color= '#7994B0', alpha=0.5)

# Adding Freemocap mean and error bars
axs[0].errorbar(conditions, freemocap_mean, yerr=freemocap_std, fmt='-o', color='black', capsize=5, label='Mean')

# Labels and titles for Freemocap
axs[0].set_title('Freemocap', fontsize = 16)
axs[0].set_ylabel('Path Length (mm)', fontsize = 14)
axs[0].legend(loc = 'upper left')
axs[0].set_xlabel('Condition', fontsize = 14)

# Plotting Qualisys data
for index, row in qualisys_path_lengths.iterrows():
    axs[1].plot(conditions, row, '-o', color= '#C67548', alpha=0.5)

# Adding Qualisys mean and error bars
axs[1].errorbar(conditions, qualisys_mean, yerr=qualisys_std, fmt='-o', color='black', capsize=5, label='Mean')

# Labels and titles for Qualisys
axs[1].set_title('Qualisys', fontsize = 16)
axs[1].set_xlabel('Condition', fontsize = 14)
axs[1].legend(loc = 'upper left')

# Display the plot
plt.tight_layout()
plt.show()