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

def calculate_joint_centers(df, joint_center_mapping):
    # Create an empty list to store the calculated joint centers
    joint_centers = []

    # Precompute mappings and values
    precomputed_values = {}
    for joint_name, marker_dimensions_and_weights in joint_center_mapping.items():
        precomputed_values[joint_name] = {}
        for marker_dimension, marker_weights in marker_dimensions_and_weights.items():
            marker_names = list(marker_weights.keys())
            weights = list(marker_weights.values())
            precomputed_values[joint_name][marker_dimension] = (marker_names, weights)

    # Iterate over frames using groupby
    for frame, marker_xyz_dataframe in track(df.groupby('frame'), description='Calculating joint centers per frame'):
        # Iterate over joint centers
        for joint_name in joint_center_mapping.keys():
            # Create a dictionary to store the calculated coordinates for this joint center
            joint_center_coordinates = {'frame': frame, 'joint_center': joint_name}

            # Iterate over dimensions (x, y, z)
            for marker_dimension in joint_center_mapping[joint_name].keys():
                marker_names, weights = precomputed_values[joint_name][marker_dimension]
                available_marker_data = marker_xyz_dataframe[marker_xyz_dataframe['marker'].isin(marker_names)]

                # Calculate the weighted sum of the coordinates for this dimension
                weighted_sum = sum(
                    available_marker_data.loc[available_marker_data['marker'] == marker_name, marker_dimension].values[0] * weight
                    for marker_name, weight in zip(marker_names, weights) if marker_name in available_marker_data['marker'].values
                )

                # Calculate the sum of the weights
                sum_weights = sum(weight for marker_name, weight in zip(marker_names, weights) if marker_name in available_marker_data['marker'].values)

                # Calculate the weighted average
                if sum_weights != 0:
                    weighted_avg = weighted_sum / sum_weights
                else:
                    weighted_avg = None

                # Store the calculated coordinate in the joint_center_coordinates dictionary
                joint_center_coordinates[marker_dimension] = weighted_avg

            # Add the calculated coordinates for this joint center to the joint_centers list
            joint_centers.append(joint_center_coordinates)

    # Convert the joint_centers list to a DataFrame
    joint_centers_df = pd.DataFrame(joint_centers)

    return joint_centers_df
