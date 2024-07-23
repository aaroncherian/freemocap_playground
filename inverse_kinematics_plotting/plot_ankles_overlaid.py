import pandas as pd
import plotly.graph_objs as go
from pathlib import Path

# List of paths to FreeMoCap folders
paths_to_freemocap_folders = [
    r'D:\2023-06-07_TF01\1.0_recordings\treadmill_calib\sesh_2023-06-07_11_55_05_TF01_flexion_neg_5_6_trial_1',
    r'D:\2023-06-07_TF01\1.0_recordings\treadmill_calib\sesh_2023-06-07_12_03_15_TF01_flexion_neg_2_8_trial_1',
    r'D:\2023-06-07_TF01\1.0_recordings\treadmill_calib\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1',
    r'D:\2023-06-07_TF01\1.0_recordings\treadmill_calib\sesh_2023-06-07_12_09_05_TF01_flexion_pos_2_8_trial_1',
    r'D:\2023-06-07_TF01\1.0_recordings\treadmill_calib\sesh_2023-06-07_12_12_36_TF01_flexion_pos_5_6_trial_1'
]

# Labels for the plots
labels = ['-5.6', '-2.8', 'neutral', '+2.8', '+5.6']

# Create a figure
fig = go.Figure()

# Loop over each path and label
for path, label in zip(paths_to_freemocap_folders, labels):
    path_to_ik_data = Path(path) / 'output_data' / 'IK_results.mot'
    
    # Read the data
    ik_data = pd.read_csv(path_to_ik_data, sep='\t', skiprows=10)
    
    # Add right ankle angle plot
    fig.add_trace(go.Scatter(
        x=ik_data['time'],
        y=ik_data['ankle_angle_r'],
        mode='lines',
        name=f'Right Ankle Angle ({label})'
    ))

# Update layout
fig.update_layout(
    title='Ankle Angles Over Time for Different Flexion Angles',
    xaxis_title='Time (s)',
    yaxis_title='Angle (degrees)',
    legend_title='Ankle Angles'
)

# Show the figure
fig.show()
