
from qualisys.calculate_joint_centers import calculate_joint_centers
import pandas as pd
from pathlib import Path

def dataframe_to_numpy(df):
    # Get the list of unique markers in the order they appear for frame 0
    marker_order = df['marker'].unique().tolist()
    
    # Create a dictionary to map marker names to their order
    marker_order_dict = {marker: idx for idx, marker in enumerate(marker_order)}
    
    # Sort DataFrame by 'frame' and then by the custom marker order
    df['marker_rank'] = df['marker'].map(marker_order_dict)
    df_sorted = df.sort_values(by=['frame', 'marker_rank']).drop(columns=['marker_rank'])
    
    # Extract the x, y, z columns as a NumPy array
    coords_array = df_sorted[['x', 'y', 'z']].to_numpy()
    
    # Get the number of unique frames and markers
    num_frames = df['frame'].nunique()
    num_markers = len(marker_order)
    
    # Reshape the array into the desired shape (frames, markers, dimensions)
    reshaped_array = coords_array.reshape((num_frames, num_markers, 3))
    
    return reshaped_array


def create_generic_qualisys_marker_dataframe(qualisys_biomechanical_marker_dataframe: pd.DataFrame, qualisys_marker_mappings):

    flat_mappings = {}
    for joint, markers in qualisys_marker_mappings.items():
        for biomechanical_name, generic_name in markers.items():
            flat_mappings[biomechanical_name] = generic_name
    
    # Filter rows to keep only the markers that are in the flat_mappings dictionary
    qualisys_generic_marker_dataframe = qualisys_biomechanical_marker_dataframe[qualisys_biomechanical_marker_dataframe['marker'].isin(flat_mappings.keys())]

    # Replace the marker names in the DataFrame
    qualisys_generic_marker_dataframe['marker'] = qualisys_generic_marker_dataframe['marker'].replace(flat_mappings)

    return qualisys_generic_marker_dataframe


def main(original_qualisys_dataframe: pd.DataFrame, joint_center_weights: dict):
    

    qualisys_generic_marker_dataframe = create_generic_qualisys_marker_dataframe(original_qualisys_dataframe, qualisys_marker_mappings)
    qualisys_markers_frame_marker_dimension = dataframe_to_numpy(qualisys_generic_marker_dataframe)
    marker_names = qualisys_generic_marker_dataframe['marker'].unique().tolist()
    joint_centers_frame_marker_dimension = calculate_joint_centers(qualisys_markers_frame_marker_dimension, joint_center_weights, marker_names)

    return joint_centers_frame_marker_dimension, qualisys_markers_frame_marker_dimension




if __name__ == '__main__':

    from qualisys.joint_center_calculation.qualisys_generic_marker_mapping import qualisys_marker_mappings
    from qualisys.qualisys_plotting import plot_3d_scatter
    from qualisys.joint_center_calculation.qualisys_joint_center_mapping import joint_center_weights

    path_to_recording_folder = Path(r"D:\2023-06-07_JH\1.0_recordings\treadmill_calib\sesh_2023-06-07_12_06_15_JH_flexion_neutral_trial_1")
    path_to_qualisys_folder = path_to_recording_folder / 'qualisys'
    path_to_qualisys_csv = path_to_qualisys_folder / 'qualisys_markers_dataframe.csv'

    qualisys_dataframe = pd.read_csv(path_to_qualisys_csv)

    joint_centers_frame_marker_dimension,qualisys_markers_frame_marker_dimension = main(qualisys_dataframe, joint_center_weights)

    data_arrays_to_plot = {
        'qualisys markers': qualisys_markers_frame_marker_dimension,
        'qualisys joint centers': joint_centers_frame_marker_dimension,
    }
    plot_3d_scatter(data_arrays_to_plot)

    f = 2
