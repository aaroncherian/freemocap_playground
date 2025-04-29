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
            opacity=.5
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

    # windows_data_path = Path(r'C:\Users\aaron\freemocap_data\recording_sessions\freemocap_test_data_321')
    # macos_data_path = Path(r'C:\Users\aaron\freemocap_data\recording_sessions\order_test_with_pin_to_0\freemocap_test_data_321')
    # # ubuntu_data_path = Path(r'C:\Users\aaron\freemocap_data\recording_sessions\freemocap_test_data_321')

    # # path_dict = {'windows_123_no_pin': windows_data_path, 'windows_231_no_pin': macos_data_path, 'windows_321_no_pin': ubuntu_data_path}
    # path_dict = {'windows_321_no_pin': windows_data_path, 'windows_321_pinned': macos_data_path,}



    # og_data = Path(r'C:\Users\aaron\freemocap_data\recording_sessions\freemocap_test_data_123_original')
    # zero_path = Path(r'C:\Users\aaron\freemocap_data\recording_sessions\freemocap_test_data_123_zero')
    # zero_fixed_path = Path(r'C:\Users\aaron\freemocap_data\recording_sessions\freemocap_test_data_123_zero_proper')
    # pos500_path = Path(r'C:\Users\aaron\freemocap_data\recording_sessions\freemocap_test_data_123_pos500')
    # neg500_path = Path(r'C:\Users\aaron\freemocap_data\recording_sessions\freemocap_test_data_123_neg500')

    # order_123 = Path(r'C:\Users\aaron\freemocap_data\recording_sessions\order_test_pin_to_0_fixed\freemocap_test_data_123')
    # order_231 = Path(r'C:\Users\aaron\freemocap_data\recording_sessions\order_test_pin_to_0_fixed\freemocap_test_data_231')
    # order_321 = Path(r'C:\Users\aaron\freemocap_data\recording_sessions\order_test_pin_to_0_fixed\freemocap_test_data_321')
    # path_dict = {'video_order_123': order_123, 'video_order_231': order_231, 'video_order_321': order_321}

    
    original_calibration = Path(r'C:\Users\aaron\freemocap_data\recording_sessions\freemocap_test_data_123_original')
    translation_zero_calibration = Path(r'C:\Users\aaron\freemocap_data\recording_sessions\freemocap_test_data_123_zero_proper')
    translation_rotation_zero_calibration = Path(r'C:\Users\aaron\freemocap_data\recording_sessions\freemocap_test_data_123_zero_rotation')
    path_dict = {'original_calibration': original_calibration, 'translation_zero_calibration': translation_zero_calibration, 'translation_rotation_zero_calibration': translation_rotation_zero_calibration}

    data_dict = {}

    for system, path_name in path_dict.items():
        data = np.load(path_name/'output_data'/'mediapipe_skeleton_3d.npy')[:,0:33,:]
        data_dict[system] = data




    # windows_data = np.load(r"D:\system_testing\no_pin_to_zero\test_data_artifacts_windows-latest (8)\output_data\mediapipe_skeleton_3d.npy") [:,0:33,:]
    # mac_data = np.load(r"D:\system_testing\no_pin_to_zero\test_data_artifacts_macos-latest (10)\output_data\mediapipe_skeleton_3d.npy")[:,0:33,:]
    # linux_data = np.load(r"D:\system_testing\no_pin_to_zero\test_data_artifacts_ubuntu-latest (9)\output_data\mediapipe_skeleton_3d.npy")[:,0:33,:]


    from skellyforge.freemocap_utils.postprocessing_widgets.postprocessing_functions.rotate_skeleton import align_skeleton_with_origin


    import numpy as np

    from skellymodels.model_info.mediapipe_model_info import MediapipeModelInfo
    

    data_aligned = {}

    for array_name, array in data_dict.items():
        good_frame = 103

        aligned_data = align_skeleton_with_origin(skeleton_data=array, skeleton_indices=MediapipeModelInfo.body_landmark_names, good_frame=good_frame)[0]   

        data_aligned[array_name] = aligned_data






    plot_3d_scatter(data_aligned)
    f = 2