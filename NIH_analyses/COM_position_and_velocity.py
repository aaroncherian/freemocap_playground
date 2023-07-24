import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

path_to_data_folder = Path(r'D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3')

data_folder = 'output_data'
center_of_mass_folder = 'center_of_mass'
total_body_com_file = 'total_body_center_of_mass_xyz.npy'
analysis_folder = 'data_analysis'

sessionIDs = ['sesh_2023-05-17_14_40_56_MDN_NIH_Trial2']

colors = ['blue', 'green', 'red']
dimensions = ['X', 'Y', 'Z']

# Create a figure and subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10), sharex=True)

# For each session, load data and plot
for j, sessionID in enumerate(sessionIDs):
    # Load center of mass data
    com_file_path = path_to_data_folder / sessionID / data_folder / center_of_mass_folder / total_body_com_file
    com_data = np.load(com_file_path)

    # Load condition_data.json
    analysis_subfolders = sorted((path_to_data_folder / sessionID / analysis_folder).glob('analysis_*'))
    if not analysis_subfolders:
        print(f"No analysis subfolders found for session {sessionID}")
        continue
    condition_json_path = analysis_subfolders[-1] / 'condition_data.json'
    with open(condition_json_path, 'r') as file:
        condition_data = json.load(file)
    
    # Extract frame intervals for 'Eyes Open/Solid Ground'
    frame_intervals = condition_data["Frame Intervals"]["Eyes Open/Solid Ground"]
    start_frame, end_frame = frame_intervals

    # Extract the x, y data for the given frames
    for i in range(2):
        # Position
        pos_data = com_data[100:10000, i]
        axes[0, i].plot(pos_data, label=f'{dimensions[i]} Position', color=colors[i])
        axes[0, i].set_xlabel('Frame #')
        axes[0, i].set_ylabel('Position (mm)')
        axes[0, i].set_title(f'Trial {j+1} COM {dimensions[i]} Position ')
        axes[0, i].legend(loc='upper left')
        
        # Velocity
        vel_data = np.diff(com_data[100:10000, i])
        axes[1, i].plot(vel_data, label=f'{dimensions[i]} Velocity', color=colors[i])
        axes[1, i].set_xlabel('Frame #')
        axes[1, i].set_ylabel('Velocity (mm/frame)')
        axes[1, i].set_title(f'Trial {j+1} COM {dimensions[i]} Velocity')
        axes[1, i].legend(loc='upper left')

# Tight layout for better spacing
plt.tight_layout()

# Show plot
plt.show()
