from nicegui import ui
import numpy as np
import plotly.graph_objects as go


from skellymodels.experimental.model_redo.managers.human import Human
from skellymodels.experimental.model_redo.tracker_info.model_info import ModelInfo
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


path_to_recording = Path(r'D:\recording_15_26_59_gmt-4__kk_bad_form')
path_to_output_data = path_to_recording / 'output_data'/ 'mediapipe_body_3d_xyz.npy'


model_info = ModelInfo(config_path = Path(__file__).parent/'mediapipe_just_body.yaml')
human = Human.from_numpy_array(
    name = 'human', 
    model_info=model_info,
    tracked_points_numpy_array=np.load(path_to_output_data)
)

from nicegui import ui
import numpy as np
import plotly.graph_objects as go

# Your data preparation code stays the same
body_3d_xyz = human.body.trajectories['3d_xyz']
spine_vector = body_3d_xyz.segment_data['spine']['proximal'] - body_3d_xyz.segment_data['spine']['distal']
spine_vector_magnitude = np.linalg.norm(spine_vector, axis=1)
spine_vector_azimuthal = np.arctan2(spine_vector[:, 1], spine_vector[:, 0])
spine_vector_polar = np.arccos(spine_vector[:,2]/(spine_vector_magnitude + 1e-9))

num_frames = body_3d_xyz.num_frames
x = spine_vector_magnitude * np.sin(spine_vector_polar) * np.cos(spine_vector_azimuthal)
y = spine_vector_magnitude * np.sin(spine_vector_polar) * np.sin(spine_vector_azimuthal)
z = spine_vector_magnitude * np.cos(spine_vector_polar)

# Calculate the axis ranges based on all frames
# Convert numpy values to Python floats for JSON serialization
x_range = [float(min(min(x), 0)), float(max(max(x), 0))]
y_range = [float(min(min(y), 0)), float(max(max(y), 0))]
z_range = [float(min(min(z), 0)), float(max(max(z), 0))]

# Create initial figure
fig = go.Figure()
fig.add_trace(go.Scatter3d(
    x=[0, float(x[0])],  # Convert to float
    y=[0, float(y[0])],  # Convert to float
    z=[0, float(z[0])],  # Convert to float
    mode='lines+markers',
    name='Spine Vector',
    marker=dict(size=5),
    line=dict(width=3)
))

fig.update_layout(
    title='Spine Vector Visualization',
    scene=dict(
        xaxis=dict(range=x_range),
        yaxis=dict(range=y_range),
        zaxis=dict(range=z_range),
        aspectmode='cube'
    ),
    showlegend=True,
    uirevision='true'  # Changed to string 'true'
)

# Create the plot element
plot = ui.plotly(fig).classes('w-[800px] h-[800px]')

# Function to update the plot
def update_plot(event):
    frame_idx = int(event.value)
    # Update the plot data while maintaining camera position
    plot.update_figure({
        'data': [
            {
                'type': 'scatter3d',
                'x': [0, float(x[frame_idx])],  # Convert to float
                'y': [0, float(y[frame_idx])],  # Convert to float
                'z': [0, float(z[frame_idx])],  # Convert to float
                'mode': 'lines+markers',
                'name': 'Spine Vector',
                'marker': {'size': 5},
                'line': {'width': 3}
            }
        ],
        'layout': {
            'scene': {
                'xaxis': {'range': x_range},
                'yaxis': {'range': y_range},
                'zaxis': {'range': z_range},
                'aspectmode': 'cube'
            },
            'uirevision': 'true'  # Changed to string 'true'
        }
    })

# Add frame counter display
frame_label = ui.label('Frame: 0')
def update_label(event):
    frame_label.text = f'Frame: {int(event.value)}'

# Update both plot and label when slider changes
def on_slider_change(event):
    update_plot(event)
    update_label(event)

# Create slider with the combined callback
ui.slider(
    min=0, 
    max=num_frames-1,
    step=1,
    value=0,
    on_change=on_slider_change
).classes('w-64')

ui.run()