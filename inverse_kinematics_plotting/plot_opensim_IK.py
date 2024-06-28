from pathlib import Path
import pandas as pd
# import plotly.graph_objs as go
# import plotly.subplots as sp


path_to_freemocap_folder = Path(r"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_13_37_32_MDN_treadmill_1")
# path_to_freemocap_folder = Path(r'D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_13_48_44_MDN_treadmill_2')
# path_to_freemocap_folder = Path(r'D:\2023-06-07_TF01\1.0_recordings\treadmill_calib\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1')
# path_to_freemocap_folder = Path(r'D:\2023-06-07_TF01\1.0_recordings\treadmill_calib\sesh_2023-06-07_12_09_05_TF01_flexion_pos_2_8_trial_1')
# path_to_freemocap_folder = Path(r'D:\2023-06-07_TF01\1.0_recordings\treadmill_calib\sesh_2023-06-07_12_12_36_TF01_flexion_pos_5_6_trial_1')

path_to_ik_data = path_to_freemocap_folder/'output_data'/'IK_results.mot'
ik_data = data = pd.read_csv(path_to_ik_data, sep='\t', skiprows=10)



# Create subplots
fig = sp.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)

# Add left ankle angle plot
fig.add_trace(go.Scatter(
    x=ik_data['time'],
    y=ik_data['ankle_angle_l'],
    mode='lines',
    name='Left Ankle Angle'
), row=1, col=1)


# Add right ankle angle plot
fig.add_trace(go.Scatter(
    x=ik_data['time'],
    y=ik_data['ankle_angle_r'],
    mode='lines',
    name='Right Ankle Angle'
), row=2, col=1)

# Update layout
fig.update_layout(
    title='Ankle Angles Over Time',
    xaxis_title='Time (s)',
    yaxis_title='Angle (degrees)',
    legend_title='Ankle Angles'
)

# Update y-axis titles
fig.update_yaxes(title_text="Left Ankle Angle (degrees)", row=1, col=1)
fig.update_yaxes(title_text="Right Ankle Angle (degrees)", row=2, col=1)

# Show the figure
fig.show()