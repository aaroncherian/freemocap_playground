import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
from pathlib import Path

# Define paths
path_to_freemocap_folder = Path(r'D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_13_48_44_MDN_treadmill_2')

path_to_qualisys_ik_data = path_to_freemocap_folder / 'output_data' / 'component_qualisys_synced' / 'IK_results.mot'
path_to_freemocap_ik_data = path_to_freemocap_folder / 'output_data' / 'IK_results.mot'

# Load data
qualisys_data = pd.read_csv(path_to_qualisys_ik_data, sep='\t', skiprows=10)
freemocap_data = pd.read_csv(path_to_freemocap_ik_data, sep='\t', skiprows=10)

# Create a frame index based on row numbers
qualisys_data['frame'] = range(len(qualisys_data))
freemocap_data['frame'] = range(len(freemocap_data))

# Create subplots
fig = sp.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)

thing_to_plot = 'ankle_angle'


# Add left ankle angle plots
fig.add_trace(go.Scatter(
    x=qualisys_data['frame'],
    y=qualisys_data[f'pelvis_tilt'],
    mode='lines',
    name='Left Ankle Angle (Qualisys)'
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=freemocap_data['frame'],
    y=freemocap_data[f'pelvis_tilt'],
    mode='lines',
    name='Left Ankle Angle (FreeMoCap)'
), row=1, col=1)

# Add right ankle angle plots
fig.add_trace(go.Scatter(
    x=qualisys_data['frame'],
    y=qualisys_data[f'{thing_to_plot}_r'],
    mode='lines',
    name='Right Ankle Angle (Qualisys)'
), row=2, col=1)

fig.add_trace(go.Scatter(
    x=freemocap_data['frame'],
    y=-freemocap_data[f'{thing_to_plot}_r'],
    mode='lines',
    name='Right Ankle Angle (FreeMoCap)'
), row=2, col=1)

# Update layout
fig.update_layout(
    title='Ankle Angles Over Frames (Qualisys vs FreeMoCap)',
    xaxis_title='Frame Index',
    legend_title='Ankle Angles'
)

# Update y-axis titles
fig.update_yaxes(title_text="Left Ankle Angle (degrees)", row=1, col=1)
fig.update_yaxes(title_text="Right Ankle Angle (degrees)", row=2, col=1)

# Show the figure
fig.show()
