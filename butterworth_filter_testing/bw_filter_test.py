from pathlib import Path
import numpy as np

og_data = np.load(Path(r"D:\2024-08-01_treadmill_KK_JSM_ATC\1.0_recordings\sesh_2024-08-01_15_51_10_KK_treadmill_2_original\output_data\raw_data\mediapipe3dData_numFrames_numTrackedPoints_spatialXYZ.npy"))
cutoff_1hz = np.load(Path(r"D:\2024-08-01_treadmill_KK_JSM_ATC\1.0_recordings\sesh_2024-08-01_15_51_10_KK_treadmill_2\output_data\mediapipe_skeleton_3d.npy"))
cutoff_7hz = np.load(Path(r"D:\2024-08-01_treadmill_KK_JSM_ATC\1.0_recordings\sesh_2024-08-01_15_51_10_KK_treadmill_2_original\output_data\mediaPipeSkel_3d_body_hands_face.npy"))
import plotly.graph_objects as go
import numpy as np

def plot_markers_dropdown(og_data, cutoff_7hz, cutoff_1hz):
    num_markers = min(33, og_data.shape[1])  # Limit to first 33 markers
    num_frames = og_data.shape[0]

    fig = go.Figure()

    # Add all markers as traces, initially hiding them
    for marker_idx in range(num_markers):
        visible = True if marker_idx == 0 else False  # Show only the first marker initially
        
        fig.add_trace(go.Scatter(
            y=og_data[:, marker_idx, 2], 
            x=np.arange(num_frames), 
            mode='lines', 
            name='Original', 
            visible=visible
        ))
        
        fig.add_trace(go.Scatter(
            y=cutoff_7hz[:, marker_idx, 2], 
            x=np.arange(num_frames), 
            mode='lines', 
            name='7 Hz Cutoff', 
            visible=visible
        ))
        
        fig.add_trace(go.Scatter(
            y=cutoff_1hz[:, marker_idx, 2], 
            x=np.arange(num_frames), 
            mode='lines', 
            name='1 Hz Cutoff', 
            visible=visible
        ))

    # Create dropdown buttons
    buttons = []
    for marker_idx in range(num_markers):
        button = dict(
            label=f"Marker {marker_idx + 1}",
            method="update",
            args=[{"visible": [False] * (num_markers * 3)}]  # Hide all traces
        )
        # Make selected marker's traces visible
        button["args"][0]["visible"][marker_idx * 3] = True  # Original
        button["args"][0]["visible"][marker_idx * 3 + 1] = True  # 7 Hz Cutoff
        button["args"][0]["visible"][marker_idx * 3 + 2] = True  # 1 Hz Cutoff

        buttons.append(button)

    # Add dropdown menu
    fig.update_layout(
        updatemenus=[{
            "buttons": buttons,
            "direction": "down",
            "showactive": True,
        }],
        title="Select Marker to Display",
        xaxis_title="Frame",
        yaxis_title="Z Position",
        template="plotly_white"
    )

    return fig

# Example usage:
fig = plot_markers_dropdown(og_data, cutoff_7hz, cutoff_1hz)
fig.show()
