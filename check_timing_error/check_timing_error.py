from pathlib import Path
from skellymodels.managers.human import Human
import numpy as np
import plotly.graph_objects as go

path_to_recording = Path(r"D:\2025_07_31_JSM_pilot\freemocap\2025-07-31_16-35-10_GMT-4_jsm_treadmill_trial_1\validation")

trackers = ["mediapipe", "qualisys", "rtmpose"]
frames_to_use = [660,1260]
tracker_dict = {}
for tracker in trackers:
    path_to_data = path_to_recording/ tracker
    human:Human = Human.from_data(path_to_data)

    data = human.body.xyz.as_dict['left_heel'][...,1]
    tracker_dict[tracker] = data

fig = go.Figure()

start, end = frames_to_use

for tracker, z_data in tracker_dict.items():
    z_data = np.asarray(z_data)

    # Optional frame window
    if frames_to_use is not None:
        z_data = z_data[start:end]
        x = np.arange(start, end)
    else:
        x = np.arange(len(z_data))

    fig.add_trace(
        go.Scatter(
            x=x,
            y=z_data,
            mode="lines",
            name=tracker,
        )
    )

fig.update_layout(
    title="Left Knee Y Position",
    xaxis_title="Frame",
    yaxis_title="Z Position (meters)",
    template="simple_white",
    legend_title="Tracker",
)

fig.show()

f = 2