import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np
from pathlib import Path
import seaborn as sns

# Helper function to aggregate path lengths
def aggregate_path_lengths(analysis_folder):
    json_path = analysis_folder / 'condition_data.json'
    
    # Check if JSON file exists
    if not json_path.exists():
        print(f"JSON file not found in {analysis_folder}.")
        return {}

    # Read the JSON data
    json_data = json.load(open(json_path))

    # Extract path lengths
    return json_data.get('Path Lengths:', {})

# Analysis folders for each trial
analysis_folders_freemocap = [
    Path(r'D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_14_40_56_MDN_NIH_Trial2\data_analysis\analysis_2023-06-01_10_03_59'),
    Path(r'D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_14_53_48_MDN_NIH_Trial3\data_analysis\analysis_2023-06-01_10_12_24'),
    Path(r'D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_15_03_20_MDN_NIH_Trial4\data_analysis\analysis_2023-06-01_10_17_22'),
]

analysis_folders_qualisys = [
    Path(r'D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\qualisys_MDN_NIH_Trial2\data_analysis\analysis_2023-06-01_16_11_00'),
    Path(r'D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\qualisys_MDN_NIH_Trial3\data_analysis\analysis_2023-06-01_17_14_40'),
    Path(r'D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\qualisys_MDN_NIH_Trial4\data_analysis\analysis_2023-06-01_18_06_59')
]

# Variables for plotting
colors = ['#4d7197', '#bf6431']
labels = ['FreeMoCap', 'Qualisys']
bar_width = 0.35
sns.set_style('whitegrid')
# Loop through each trial
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6), sharey=True)

# Loop through each trial
for i, (folder_freemocap, folder_qualisys, ax) in enumerate(zip(analysis_folders_freemocap, analysis_folders_qualisys, axes)):
    
    # Get path lengths
    path_lengths_freemocap = aggregate_path_lengths(Path(folder_freemocap))
    path_lengths_qualisys = aggregate_path_lengths(Path(folder_qualisys))
    conditions = list(path_lengths_freemocap.keys())
    x = np.arange(len(conditions))  # the label locations
    
    # Data for plotting
    data_freemocap = list(path_lengths_freemocap.values())
    data_qualisys = list(path_lengths_qualisys.values())

    # Plot bars with path length values
    for j, data in enumerate([data_freemocap, data_qualisys]):
        ax.bar(x + j * bar_width, data, width=bar_width, color=colors[j], alpha=0.8)
        for k, value in enumerate(data):
            ax.text(k + j * bar_width, value, f'{value:.2f}', ha='center', va='bottom')
    
    # Adding labels and title
    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels(conditions)
    ax.set_xlabel('Condition', fontsize=12, fontweight='bold')
    if i == 0:
        ax.set_ylabel('Path Length (mm)', fontsize=12, fontweight='bold')
    ax.set_title(f'Trial {i + 1} Path Length Comparison', fontsize=14, fontweight='bold')
    # plt.xticks(rotation=10)

# Add a single legend for the entire figure
fig.legend(labels, title='System', loc='upper right')

# Remove spines
sns.despine(left=True, fig=fig)

# Show plots
plt.show()


# Remove spines
sns.despine(left=True, fig=fig)

# Show plots
plt.show()
    # plt.savefig(save_path)

#
