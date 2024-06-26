from pathlib import Path
import pandas as pd
import plotly.graph_objs as go
import plotly.subplots as sp

# List of paths to FreeMoCap folders
paths_to_freemocap_folders = [
    r'D:\2023-06-07_TF01\1.0_recordings\treadmill_calib\sesh_2023-06-07_11_55_05_TF01_flexion_neg_5_6_trial_1',
    r'D:\2023-06-07_TF01\1.0_recordings\treadmill_calib\sesh_2023-06-07_12_03_15_TF01_flexion_neg_2_8_trial_1',
    r'D:\2023-06-07_TF01\1.0_recordings\treadmill_calib\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1',
    r'D:\2023-06-07_TF01\1.0_recordings\treadmill_calib\sesh_2023-06-07_12_09_05_TF01_flexion_pos_2_8_trial_1',
    r'D:\2023-06-07_TF01\1.0_recordings\treadmill_calib\sesh_2023-06-07_12_12_36_TF01_flexion_pos_5_6_trial_1'
]

# Labels for the subplots
labels = ['-5.6', '-2.8', 'neutral', '+2.8', '+5.6']

# Create subplots
fig = sp.make_subplots(rows=len(paths_to_freemocap_folders), cols=1, shared_xaxes=True, vertical_spacing=0.05)

# Loop over each path and label
for i, (path, label) in enumerate(zip(paths_to_freemocap_folders, labels)):
    path_to_ik_data = Path(path) / 'output_data' / 'IK_results.mot'
    
    # Read the data
    ik_data = pd.read_csv(path_to_ik_data, sep='\t', skiprows=10)
    
    # Add right ankle angle plot
    fig.add_trace(go.Scatter(
        x=ik_data['time'],
        y=ik_data['ankle_angle_r'],
        mode='lines',
        name=f'Right Ankle Angle ({label})'
    ), row=i+1, col=1)
    
    # # Add left ankle angle plot
    # fig.add_trace(go.Scatter(
    #     x=ik_data['time'],
    #     y=ik_data['ankle_angle_l'],
    #     mode='lines',
    #     name=f'Left Ankle Angle ({label})'
    # ), row=i+1, col=1)
    
    # Update y-axis title for each subplot
    fig.update_yaxes(title_text=f'Angle (degrees) ({label})', row=i+1, col=1)

# Set the same y-axis range for all subplots
y_axis_range = [-15, 25]  # Adjust this range based on your data

for i in range(1, len(paths_to_freemocap_folders) + 1):
    fig.update_yaxes(range=y_axis_range, row=i, col=1)

# Update layout
fig.update_layout(
    title='Ankle Angles Over Time for Different Flexion Angles',
    xaxis_title='Time (s)',
    legend_title='Ankle Angles'
)

# Show the figure
fig.show()
