import pandas as pd
from rich.progress import track

# def calculate_joint_centers(df, joint_center_mapping):
#     # Create an empty list to store the calculated joint centers
#     joint_centers = []

#     df_sorted = df.sort_values(by='frame')
#     # Iterate over frames
#     for frame,marker_xyz_dataframe in track(df_sorted.groupby('frame'), description='Calculating joint centers per frame'):
#         # Filter the DataFrame for the current frame
#         # marker_xyz_dataframe = df[df['frame'] == frame]

#         # Iterate over joint centers
#         for joint_name, marker_dimensions_and_weights in joint_center_mapping.items():
#             # Create a dictionary to store the calculated coordinates for this joint center
#             joint_center_coordinates = {'frame': frame, 'joint_center': joint_name}

#             # Iterate over dimensions (x, y, z)
#             for marker_dimension, marker_weights in marker_dimensions_and_weights.items():
#                 # Calculate the weighted sum of the coordinates for this dimension
#                 weighted_sum = sum(marker_xyz_dataframe.loc[marker_xyz_dataframe['marker'] == marker_name, marker_dimension].values[0] * weight
#                                    for marker_name, weight in marker_weights.items() if marker_name in marker_xyz_dataframe['marker'].values) #iterate over each marker and its associated weight, but only if that marker exists in the current frame

#                 # Calculate the sum of the weights
#                 sum_weights = sum(weight for marker_name, weight in marker_weights.items() if marker_name in marker_xyz_dataframe['marker'].values)

#                 # Calculate the weighted average
#                 if sum_weights != 0:
#                     weighted_avg = weighted_sum / sum_weights
#                 else:
#                     weighted_avg = None

#                 # Store the calculated coordinate in the joint_center_coordinates dictionary
#                 joint_center_coordinates[marker_dimension] = weighted_avg

#             # Add the calculated coordinates for this joint center to the joint_centers list
#             joint_centers.append(joint_center_coordinates)

#     # Convert the joint_centers list to a DataFrame
#     joint_centers_df = pd.DataFrame(joint_centers)

#     return joint_centers_df


# def calculate_joint_centers(df, joint_center_mapping):
#     # Create an empty list to store the calculated joint centers
#     joint_centers = []
    
#     # Pre-compute mappings and values
#     precomputed_values = {}
#     for joint_name, marker_dimensions_and_weights in track(joint_center_mapping.items(), description='precomputing'):
#         precomputed_values[joint_name] = {}
#         for marker_dimension, marker_weights in marker_dimensions_and_weights.items():
#             marker_names = list(marker_weights.keys())
#             weights = list(marker_weights.values())
#             precomputed_values[joint_name][marker_dimension] = (marker_names, weights)
    
#     # Iterate over frames
#     for frame in track(df['frame'].unique(), description='calculating per frame'):
#         # Filter the DataFrame for the current frame
#         marker_xyz_dataframe = df[df['frame'] == frame]
        
#         # Iterate over joint centers
#         for joint_name in joint_center_mapping.keys():
#             # Create a dictionary to store the calculated coordinates for this joint center
#             joint_center_coordinates = {'frame': frame, 'joint_center': joint_name}

#             # Iterate over dimensions (x, y, z)
#             for marker_dimension in joint_center_mapping[joint_name].keys():
#                 marker_names, weights = precomputed_values[joint_name][marker_dimension]
#                 available_marker_names = [m for m in marker_names if m in marker_xyz_dataframe['marker'].values]
                
#                 # Calculate the weighted sum of the coordinates for this dimension
#                 weighted_sum = sum(
#                     marker_xyz_dataframe.loc[marker_xyz_dataframe['marker'] == marker_name, marker_dimension].values[0] * weight
#                     for marker_name, weight in zip(available_marker_names, weights)
#                 )
                
#                 # Calculate the sum of the weights
#                 sum_weights = sum(weights)

#                 # Calculate the weighted average
#                 if sum_weights != 0:
#                     weighted_avg = weighted_sum / sum_weights
#                 else:
#                     weighted_avg = None

#                 # Store the calculated coordinate in the joint_center_coordinates dictionary
#                 joint_center_coordinates[marker_dimension] = weighted_avg

#             # Add the calculated coordinates for this joint center to the joint_centers list
#             joint_centers.append(joint_center_coordinates)

#     # Convert the joint_centers list to a DataFrame
#     joint_centers_df = pd.DataFrame(joint_centers)

#     return joint_centers_df

# import numpy as np
# from rich.progress import track

# import numpy as np


# def calculate_joint_centers(df, joint_center_weights):
#     # Create a dictionary to map marker names to indices
#     unique_markers = df['marker'].unique()
#     marker_to_idx = {marker: idx for idx, marker in enumerate(unique_markers)}

#     # Convert DataFrame to 3D NumPy array [frame, marker, dimension]
#     unique_frames = df['frame'].unique()
#     num_frames = len(unique_frames)
#     num_markers = len(unique_markers)
#     num_dims = 3  # x, y, z

#     data_array = np.full((num_frames, num_markers, num_dims), np.nan)

#     for frame in track(unique_frames):
#         frame_data = df[df['frame'] == frame]
#         frame_idx = np.where(unique_frames == frame)[0][0]
        
#         for _, row in frame_data.iterrows():
#             marker_idx = marker_to_idx[row['marker']]
#             data_array[frame_idx, marker_idx, :] = [row['x'], row['y'], row['z']]

#     # Initialize an empty list to store the calculated joint centers
#     joint_centers = []

#     # Loop through unique frames
#     for frame_idx, frame in track(enumerate(unique_frames)):
#         # Loop through each joint center to be calculated
#         for joint, dimensions in joint_center_weights.items():
#             joint_center = {'frame': frame, 'joint_center': joint}

#             # Loop through each dimension (x, y, z)
#             for dim_idx, (dim, markers) in enumerate(dimensions.items()):
#                 weighted_sum = 0
#                 sum_weights = 0

#                 # Loop through each marker contributing to this dimension
#                 for marker, weight in markers.items():
#                     marker_idx = marker_to_idx.get(marker, None)
#                     if marker_idx is not None:
#                         coord_value = data_array[frame_idx, marker_idx, dim_idx]
#                         if not np.isnan(coord_value):
#                             weighted_sum += coord_value * weight
#                             sum_weights += weight

#                 # Calculate the weighted average
#                 joint_center[dim] = weighted_sum / sum_weights if sum_weights != 0 else None

#             # Append the calculated joint center to the list
#             joint_centers.append(joint_center)

#     return pd.DataFrame(joint_centers)

import pandas as pd
from rich.progress import track

# def calculate_joint_centers(df, joint_center_weights):
#     # Create an empty list to store the calculated joint centers
#     joint_centers = []
    
#     # Precompute mappings and values
#     precomputed_values = {}
#     for joint_name, marker_weights in joint_center_weights.items():
#         precomputed_values[joint_name] = {}
#         for marker_name, weights in marker_weights.items():
#             precomputed_values[joint_name][marker_name] = weights
    
#     # Iterate over frames using groupby
#     for frame, marker_xyz_dataframe in track(df.groupby('frame'), description='Calculating joint centers per frame'):
        
#         # Iterate over joint centers
#         for joint_name in joint_center_weights.keys():
            
#             # Create a dictionary to store the calculated coordinates for this joint center
#             joint_center_coordinates = {'frame': frame, 'joint_center': joint_name, 'x': 0, 'y': 0, 'z': 0}
            
#             # Initialize sums of weights for each dimension
#             sum_weights = {'x': 0, 'y': 0, 'z': 0}
            
#             # Iterate over markers and their weights
#             for marker_name, weights in precomputed_values[joint_name].items():
                
#                 if marker_name in marker_xyz_dataframe['marker'].values:
#                     marker_data = marker_xyz_dataframe[marker_xyz_dataframe['marker'] == marker_name].iloc[0]
                    
#                     for dim_idx, dimension in enumerate(['x', 'y', 'z']):
#                         weight = weights[dim_idx]
#                         joint_center_coordinates[dimension] += marker_data[dimension] * weight
#                         sum_weights[dimension] += weight
            
#             # Normalize the coordinates by the sum of weights
#             for dimension in ['x', 'y', 'z']:
#                 if sum_weights[dimension] != 0:
#                     joint_center_coordinates[dimension] /= sum_weights[dimension]
#                 else:
#                     joint_center_coordinates[dimension] = None
            
#             # Add the calculated coordinates for this joint center to the joint_centers list
#             joint_centers.append(joint_center_coordinates)
    
#     # Convert the joint_centers list to a DataFrame
#     joint_centers_df = pd.DataFrame(joint_centers)
    
#     return joint_centers_df

# # Example usage
# df: your input DataFrame
# joint_center_weights: your joint center to marker mapping
# joint_centers_df = calculate_joint_centers(df, joint_center_weights)


# import pandas as pd
# from rich.progress import track
# import numpy as np
# import time

# def calculate_frame_joint_center(marker_data, joint_to_marker_map):
#     # Drop the 'frame' level of the MultiIndex

#     marker_data = marker_data.droplevel('frame')
    
#     frame_joint_centers = {}
#     for joint, markers in joint_to_marker_map.items():
#         # Now the indexing should work without the frame level
#         weighted_positions = np.array([marker_data.loc[marker] * weight for marker, weight in markers])
#         frame_joint_centers[joint] = np.sum(weighted_positions, axis=0)
        
#     return frame_joint_centers



# def calculate_joint_centers(df, joint_center_weights):
#     # Convert the DataFrame to a multi-index format for faster lookups
#     df_multi_index = df.set_index(['frame', 'marker'])

#     total_frames = df_multi_index.index.get_level_values('frame').nunique()

#     # Function to print the current frame
#     def print_frame(frame, total_frames, start_time):
#         if frame % 1000 == 0:  # Print only every 1000 frames
#             elapsed_time = time.time() - start_time  # Calculate elapsed time in seconds
#             print(f"Finished frame {frame} of {total_frames}. Elapsed time: {elapsed_time:.2f} seconds")


#     # Modified apply function
#     def apply_with_print(frame_data, *args, **kwargs):
#         frame = frame_data.index.get_level_values('frame')[0]  # get the frame number from the MultiIndex
#         result = calculate_frame_joint_center(frame_data, *args, **kwargs)
#         print_frame(frame, total_frames, start_time)
#         return result

#     # Precompute the joint-to-marker map (same as in your previous code)
#     start_time = time.time()
#     joint_to_marker_map = {}
#     for joint, markers_for_this_joint in joint_center_weights.items():
#         joint_to_marker_map[joint] = [(marker, weight) for marker, weight in markers_for_this_joint.items()]

#     # Group by frame and apply the calculate_frame_joint_center function
#     joint_centers = df_multi_index.groupby('frame').apply(apply_with_print, joint_to_marker_map=joint_to_marker_map)

#     return joint_centers

import numpy as np

def calculate_joint_centers(array_3d, joint_center_weights, marker_names):
    num_frames, num_markers, _ = array_3d.shape
    num_joints = len(joint_center_weights.keys())
    
    # Initialize an array to hold the joint centers
    joint_centers = np.zeros((num_frames, num_joints, 3))

    # Create a mapping from marker names to indices
    marker_to_index = {marker: i for i, marker in enumerate(marker_names)}
    
    # start_time = time.time()

    # Iterate over frames
    for frame in track(range(num_frames)):
        # if frame % 1000 == 0:
        #     elapsed_time = time.time() - start_time
        #     print(f"Finished frame {frame} of {num_frames}. Elapsed time: {elapsed_time:.2f} seconds")

        # Iterate over joints
        for j_idx, joint in enumerate(joint_center_weights.keys()):
            weighted_positions = []
            for marker, weight in joint_center_weights[joint].items():
                marker_idx = marker_to_index[marker]
                weighted_positions.append(array_3d[frame, marker_idx, :] * weight)
            
            # Sum along the 0-axis to get the joint center for this frame and joint
            joint_centers[frame, j_idx, :] = np.sum(weighted_positions, axis=0)
    
    return joint_centers

    f = 2

# Example usage
# df: your input DataFrame
# joint_center_weights: your joint center to marker mapping
# joint_centers_df = calculate_joint_centers(df, joint_center_weights)
