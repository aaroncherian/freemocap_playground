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

body_3d_xyz = human.body.trajectories['3d_xyz']

spine_vector = body_3d_xyz.segment_data['spine']['proximal'] - body_3d_xyz.segment_data['spine']['distal']

spine_vector_magnitude = np.linalg.norm(spine_vector, axis=1)

spine_vector_azimuthal = np.arctan2(spine_vector[:, 1], spine_vector[:, 0])

spine_vector_polar = np.arccos(spine_vector[:,2]/(spine_vector_magnitude + 1e-9)) #1e-9 is just to prevent divide by zero errors
f = 2 

# Plot Trunk Inclination Over Time
plt.figure(figsize=(10, 5))
plt.plot(np.degrees(spine_vector_polar), label=r'Trunk Inclination $\phi$', color='b', linewidth=2)
plt.xlabel("Frame Number")
plt.ylabel("Inclination Angle (Degrees)")
plt.title("Trunk Inclination Over Time")
plt.legend()
plt.grid(True)
plt.show()

# Plot Trunk Rotation Over Time
plt.figure(figsize=(10, 5))
plt.plot(np.degrees(spine_vector_azimuthal), label=r'Trunk Rotation $\theta$', color='r', linewidth=2)
plt.xlabel("Frame Number")
plt.ylabel("Rotation Angle (Degrees)")
plt.title("Trunk Rotation Over Time")
plt.legend()
plt.grid(True)
plt.show()

# cartesian_spine_coordinates = np.stack([
#     spine_vector_magnitude * np.sin(spine_vector_polar) * np.cos(spine_vector_azimuthal),
#     spine_vector_magnitude * np.sin(spine_vector_polar) * np.sin(spine_vector_azimuthal),
#     spine_vector_magnitude  * np.cos(spine_vector_polar)
# ])
num_frames = body_3d_xyz.num_frames
x = spine_vector_magnitude * np.sin(spine_vector_polar) * np.cos(spine_vector_azimuthal)
y = spine_vector_magnitude * np.sin(spine_vector_polar) * np.sin(spine_vector_azimuthal)
z = spine_vector_magnitude * np.cos(spine_vector_polar)



# # plt.show()
# import plotly.graph_objects as go
# import plotly.io as pio

# fig = go.Figure()

# # Generate frames for animation
# frames = []
# for i in range(num_frames):
#     frame = go.Scatter3d(
#         x=[0, x[i]],
#         y=[0, y[i]],
#         z=[0, z[i]],
#         mode="lines+markers",
#         line=dict(color="blue", width=5),
#         marker=dict(size=5, color="red"),
#         name=f"Frame {i}"
#     )
#     frames.append(go.Frame(data=[frame], name=str(i)))

# # Initialize figure with the first frame
# fig.add_trace(go.Scatter3d(
#     x=[0, x[0]],
#     y=[0, y[0]],
#     z=[0, z[0]],
#     mode="lines+markers",
#     line=dict(color="blue", width=5),
#     marker=dict(size=5, color="red"),
# ))

# # Update layout for animation controls
# fig.update_layout(
#     title="3D Trunk Orientation Animation",
#     scene=dict(
#         xaxis_title="X-axis (Side-to-Side)",
#         yaxis_title="Y-axis (Forward-Backward)",
#         zaxis_title="Z-axis (Vertical)",
#         xaxis=dict(range=[-1, 1]),
#         yaxis=dict(range=[-1, 1]),
#         zaxis=dict(range=[-1, 1]),
#     ),
#     updatemenus=[dict(
#         type="buttons",
#         showactive=False,
#         buttons=[
#             dict(label="Play",
#                  method="animate",
#                  args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)]),
#             dict(label="Pause",
#                  method="animate",
#                  args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
#         ],
#     )],
#     sliders=[dict(
#         steps=[dict(method="animate", args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True))],
#                     label=str(k)) for k in range(num_frames)],
#         active=0,
#         currentvalue=dict(prefix="Frame: "),
#     )]
# )

# # Add frames to the figure
# fig.frames = frames

# # Show interactive plot
# pio.show(fig)

import plotly.graph_objects as go
import plotly.io as pio

# Create 3D figure with a reference hemisphere
# plt.show()
import plotly.graph_objects as go
import plotly.io as pio

fig = go.Figure()

# Generate frames for animation
frames = []
for i in range(num_frames):
    frame = go.Scatter3d(
        x=[0, x[i]],
        y=[0, y[i]],
        z=[0, z[i]],
        mode="lines+markers",
        line=dict(color="blue", width=5),
        marker=dict(size=5, color="red"),
        name=f"Frame {i}"
    )
    frames.append(go.Frame(data=[frame], name=str(i)))

# Initialize figure with the first frame
fig.add_trace(go.Scatter3d(
    x=[0, x[0]],
    y=[0, y[0]],
    z=[0, z[0]],
    mode="lines+markers",
    line=dict(color="blue", width=5),
    marker=dict(size=5, color="red"),
))

# Update layout for animation controls
fig.update_layout(
    title="3D Trunk Orientation Animation",
    scene=dict(
        xaxis_title="X-axis (Side-to-Side)",
        yaxis_title="Y-axis (Forward-Backward)",
        zaxis_title="Z-axis (Vertical)",
        xaxis=dict(range=[-1, 1]),
        yaxis=dict(range=[-1, 1]),
        zaxis=dict(range=[-1, 1]),
    ),
    updatemenus=[dict(
        type="buttons",
        showactive=False,
        buttons=[
            dict(label="Play",
                 method="animate",
                 args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)]),
            dict(label="Pause",
                 method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
        ],
    )],
    sliders=[dict(
        steps=[dict(method="animate", args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True))],
                    label=str(k)) for k in range(num_frames)],
        active=0,
        currentvalue=dict(prefix="Frame: "),
    )]
)

# Add frames to the figure
fig.frames = frames

# Show interactive plot
pio.show(fig)

f = 2