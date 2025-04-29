import plotly.graph_objects as go
import numpy as np

def plot_3d_scatter(data_3d_dict: dict):
    # Determine axis limits based on the data
    all_data = np.concatenate(list(data_3d_dict.values()), axis=1)
    
    mean_x = np.nanmean(all_data[:, :, 0])
    mean_y = np.nanmean(all_data[:, :, 1])
    mean_z = np.nanmean(all_data[:, :, 2])

    ax_range = 2000


    # Create a Plotly figure
    fig = go.Figure()

    # Generate a frame for each time step
    frames = []
    for frame in range(all_data.shape[0]):
        frame_data = []
        for label, data in data_3d_dict.items():
            frame_data.append(go.Scatter3d(
                x=data[frame, :, 0],
                y=data[frame, :, 1],
                z=data[frame, :, 2],
                mode='markers',
                name=label,
                marker=dict(size=4, opacity=0.8)
            ))
        frames.append(go.Frame(data=frame_data, name=str(frame)))

    # Add the first frame's data
    for label, data in data_3d_dict.items():
        fig.add_trace(go.Scatter3d(
            x=data[0, :, 0],
            y=data[0, :, 1],
            z=data[0, :, 2],
            mode='markers',
            name=label,
            marker=dict(size=4, opacity=0.8),
            opacity=.7
        ))

    # Update the layout with sliders and other settings
    fig.update_layout(
        scene=dict(
            aspectmode='cube',
            xaxis=dict(range=[mean_x - ax_range, mean_x + ax_range], title='X'),
            yaxis=dict(range=[mean_y - ax_range, mean_y + ax_range], title='Y'),
            zaxis=dict(range=[mean_z - ax_range, mean_z + ax_range], title='Z')
        ),
        title="3D Scatter Plot",
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}]),
                     dict(label="Pause",
                          method="animate",
                          args=[[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}])]
        )],
        sliders=[{
            "steps": [{"args": [[str(frame)], {"frame": {"duration": 100, "redraw": True}, "mode": "immediate"}],
                       "label": str(frame), "method": "animate"} for frame in range(all_data.shape[0])],
            "currentvalue": {"prefix": "Frame: "}
        }]
    )

    # Add the frames to the figure
    fig.frames = frames

    # Show the plot
    fig.show()


if __name__ == '__main__':
    from pathlib import Path

    freemocap_data = Path(r"D:\2025-03-13_JSM_pilot\freemocap_data\2025-03-13T16_20_37_gmt-4_pilot_jsm_treadmill_walking\validation\freemocap_3d_xyz.npy")
    qualisys_data = Path(r"D:\2025-03-13_JSM_pilot\freemocap_data\2025-03-13T16_20_37_gmt-4_pilot_jsm_treadmill_walking\validation\qualisys_3d_xyz.npy")

    # path_dict = {'windows': windows_data_path, 'macos': macos_data_path, 'ubuntu': ubuntu_data_path}

    data_dict = {}

    data_dict['freemocap'] = np.load(freemocap_data)[0:2788]
    data_dict['qualisys'] = np.load(qualisys_data)
 





    # windows_data = np.load(r"D:\system_testing\no_pin_to_zero\test_data_artifacts_windows-latest (8)\output_data\mediapipe_skeleton_3d.npy") [:,0:33,:]
    # mac_data = np.load(r"D:\system_testing\no_pin_to_zero\test_data_artifacts_macos-latest (10)\output_data\mediapipe_skeleton_3d.npy")[:,0:33,:]
    # linux_data = np.load(r"D:\system_testing\no_pin_to_zero\test_data_artifacts_ubuntu-latest (9)\output_data\mediapipe_skeleton_3d.npy")[:,0:33,:]


    from skellyforge.freemocap_utils.postprocessing_widgets.postprocessing_functions.rotate_skeleton import align_skeleton_with_origin
    from skellyforge.freemocap_utils.postprocessing_widgets.postprocessing_functions.good_frame_finder import find_good_frame


    import numpy as np

    from skellymodels.model_info.mediapipe_model_info import MediapipeModelInfo
    
    joint_center_weights = {
    'head': {
        'LHead': [.5, .5, .5],
        'RHead': [.5, .5, .5],
    },

    'left_ear': {
        'LHead': [1, 1, 1],
    },

    'right_ear': {
        'RHead': [1, 1, 1],
    },

    'left_shoulder': {
        'LFrontShoulder': [.5, .5, .5],
        'LBackShoulder': [.5, .5, .5],
    },

    'right_shoulder': {
        'RFrontShoulder': [.5, .5, .5],
        'RBackShoulder': [.5, .5, .5],
    },

    'left_elbow': {
        'LLatElbow': [.5, .5, .5],
        'LMedElbow': [.5, .5, .5],
    },

    'right_elbow': {
        'RLatElbow': [.5, .5, .5],
        'RMedElbow': [.5, .5, .5],
    },

    'left_wrist': {
        'LLatWrist': [.5, .5, .5],
        'LMedWrist': [.5, .5, .5],
    },

    'right_wrist': {
        'RLatWrist': [.5, .5, .5],
        'RMedWrist': [.5, .5, .5],
    },

    'left_hand': {
        'LHand': [1, 1, 1],
    },

    'right_hand': {
        'RHand': [1, 1, 1],
    },

    'left_hip': {
        'LIC': [.25, .25, .25],
        'LPSIS': [.25, .25, .25],
        'LASIS': [.25, .25, .25],
        'LGT': [.25, .25, .25],
    },

    'right_hip': {
        'RIC': [.25, .25, .25],
        'RPSIS': [.25, .25, .25],
        'RASIS': [.25, .25, .25],
        'RGT': [.25, .25, .25],
    },

    'left_knee': {
        'LLFC': [.5, .5, .5],
        'LMFC': [.5, .5, .5],
    },

    'right_knee': {
        'RLFC': [.5, .5, .5],
        'RMFC': [.5, .5, .5],
    },

    'left_ankle': {
        'LLMA': [.5, .5, .5],
        'LMMA': [.5, .5, .5],
    },

    'right_ankle': {
        'RLMA': [.5, .5, .5],
        'RMMA': [.5, .5, .5],
    },

    'left_heel': {
        'LHeel': [.34, .34, .34],
        'LLatHeel': [.33, .33, .33],
        'LMedHeel': [.33, .33, .33],
    },

    'right_heel': {
        'RHeel': [.34, .34, .34],
        'RLatHeel': [.33, .33, .33],
        'RMedHeel': [.33, .33, .33],
    },

    'left_foot_index': {
        'L1ST': [.34, .34, .34],
        'L5TH': [.33, .33, .33],
        'LTOE': [.33, .33, .33],
    },

    'right_foot_index': {
        'R1ST': [.34, .34, .34],
        'R5TH': [.33, .33, .33],
        'RTOE': [.33, .33, .33],
    },
    }

    data_aligned = {}

    good_frame = find_good_frame(skeleton_data=data_dict['freemocap'], skeleton_indices=MediapipeModelInfo.body_landmark_names, initial_velocity_guess=.2)

    data_aligned['freemocap'] = align_skeleton_with_origin(skeleton_data=data_dict['freemocap'], skeleton_indices=MediapipeModelInfo.body_landmark_names, good_frame=good_frame)[0]

    data_aligned['qualisys'] = align_skeleton_with_origin(skeleton_data=data_dict['qualisys'], skeleton_indices=list(joint_center_weights.keys()), good_frame=good_frame)[0]

    # for array_name, array in data_dict.items():
    #     good_frame = find_good_frame(skeleton_data=array, skeleton_indices= list(joint_center_weights.keys()))

    #     aligned_data = align_skeleton_with_origin(skeleton_data=array, skeleton_indices=MediapipeModelInfo.body_landmark_names, good_frame=good_frame)[0]   

    #     data_aligned[array_name] = aligned_data






    plot_3d_scatter(data_aligned)
    f = 2