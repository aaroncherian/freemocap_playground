mediapipe_body = [
            "nose",
            "left_eye_inner",
            "left_eye",
            "left_eye_outer",
            "right_eye_inner",
            "right_eye",
            "right_eye_outer",
            "left_ear",
            "right_ear",
            "mouth_left",
            "mouth_right",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_pinky",
            "right_pinky",
            "left_index",
            "right_index",
            "left_thumb",
            "right_thumb",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
            "left_heel",
            "right_heel",
            "left_foot_index",
            "right_foot_index",
            ]

segment_connections = {
"head":{
    'proximal': 'left_ear',
    'distal': 'right_ear'
},
"neck":{
    'proximal': 'head_center',
    'distal': 'neck_center',
},
"spine":{
    'proximal': 'neck_center',
    'distal': 'hips_center',
},
"right_shoulder":{
    "proximal": "neck_center",
    "distal": "right_shoulder"
},
"left_shoulder":{
    "proximal": "neck_center",
    "distal": "left_shoulder"
},
"right_upper_arm":{
    "proximal": "right_shoulder",
    "distal": "right_elbow"
},
"left_upper_arm":{
    "proximal": "left_shoulder",
    "distal": "left_elbow"
},
"right_forearm":{
    "proximal": "right_elbow",
    "distal": "right_wrist"
},
"left_forearm":{
    "proximal": "left_elbow",
    "distal": "left_wrist"
},
"right_hand":{
    "proximal": "right_wrist",
    "distal": "right_index"
},
"left_hand":{
    "proximal": "left_wrist",
    "distal": "left_index"
},
"right_pelvis": {
    "proximal": "hips_center",
    "distal": "right_hip"
  },
"left_pelvis": {
        "proximal": "hips_center",
        "distal": "left_hip"
    },
"right_thigh":{
    "proximal": "right_hip",
    "distal": "right_knee"
},
"left_thigh":{
    "proximal": "left_hip",
    "distal": "left_knee"
},
"right_shank":{
    "proximal": "right_knee",
    "distal": "right_ankle"
 },
"left_shank":{
    "proximal": "left_knee",
    "distal": "left_ankle"
},
"right_foot":{
    "proximal": "right_ankle",
    "distal": "right_foot_index"
},
"left_foot":{
    "proximal": "left_ankle",
    "distal": "left_foot_index"
},
"right_heel":{
    "proximal": "right_ankle",
    "distal": "right_heel"
},
"left_heel":{
    "proximal": "left_ankle",
    "distal": "left_heel"
},
"right_foot_bottom":{
    "proximal": "right_heel",
    "distal": "right_foot_index"
},
"left_foot_bottom":{
    "proximal": "left_heel",
    "distal": "left_foot_index"
},
}

virtual_markers = {
        "head_center": {
        "marker_names": ["left_ear", "right_ear"],
        "marker_weights": [0.5, 0.5],
    },
    "neck_center": {
        "marker_names": ["left_shoulder", "right_shoulder"],
        "marker_weights": [0.5, 0.5],
    },
    "trunk_center": {
        "marker_names": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
        "marker_weights": [0.25, 0.25, 0.25, 0.25],
    },
    "hips_center": {
        "marker_names": ["left_hip", "right_hip" ],
        "marker_weights": [0.5, 0.5],
    },
}



center_of_mass_anthropometric_data = {
    "head": {
        "segment_com_length": .5,
        "segment_com_percentage": 0.081,
    },
    "spine": {
        "segment_com_length": .5,
        "segment_com_percentage": 0.497,
    },
    "right_upper_arm": {
        "segment_com_length": .436,
        "segment_com_percentage": 0.028,
    },
    "left_upper_arm": {
        "segment_com_length": .436,
        "segment_com_percentage": 0.028,
    },
    "right_forearm": {
        "segment_com_length": .430,
        "segment_com_percentage": 0.016,
    },
    "left_forearm": {
        "segment_com_length": .430,
        "segment_com_percentage": 0.016,
    },
    "right_hand": {
        "segment_com_length": .506,
        "segment_com_percentage": 0.006,
    },
    "left_hand": {
        "segment_com_length": .506,
        "segment_com_percentage": 0.006,
    },
    "right_thigh": {
        "segment_com_length": .433,
        "segment_com_percentage": 0.1,
    },
    "left_thigh": {
        "segment_com_length": .433,
        "segment_com_percentage": 0.1,
    },
    "right_shank": {
        "segment_com_length": .433,
        "segment_com_percentage": 0.0465,
    },
    "left_shank": {
        "segment_com_length": .433,
        "segment_com_percentage": 0.0465,
    },
    "right_foot": {
        "segment_com_length": .5,
        "segment_com_percentage": 0.0145,
    },
    "left_foot": {
        "segment_com_length": .5,
        "segment_com_percentage": 0.0145,
    },
}



from pydantic import BaseModel, validator, root_validator
import numpy as np
from typing import Dict, List
from pathlib import Path
class Markers(BaseModel):
    markers: List[str]

class VirtualMarkers(BaseModel):
    virtual_markers: Dict[str, Dict[str, List]]

    @validator('virtual_markers', each_item=True)
    def validate_virtual_marker(cls, virtual_marker):
        marker_names = virtual_marker.get('marker_names', [])
        marker_weights = virtual_marker.get('marker_weights', [])

        if len(marker_names) != len(marker_weights):
            raise ValueError(f'The number of marker names must match the number of marker weights for {virtual_marker}. Currently there are {len(marker_names)} names and {len(marker_weights)} weights.')

        if not isinstance(marker_names, list) or not all(isinstance(name, str) for name in marker_names):
            raise ValueError(f'Marker names must be a list of strings for {marker_names}.')

        if not isinstance(marker_weights, list) or not all(isinstance(weight, (int, float)) for weight in marker_weights):
            raise ValueError(f'Marker weights must be a list of numbers for {virtual_marker}.')

        weight_sum = sum(marker_weights)
        if not 0.99 <= weight_sum <= 1.01:  # Allowing a tiny bit of floating-point leniency
            raise ValueError(f'Marker weights must sum to approximately 1 for {virtual_marker} Current sum is {weight_sum}.')

        return virtual_marker

class Segment(BaseModel):
    proximal: str
    distal: str

class Segments(BaseModel):
    markers: Markers
    virtual_markers: VirtualMarkers
    segment_connections: Dict[str, Segment]

    @root_validator
    def check_that_all_markers_exist(cls, values):
        markers = values.get('markers').markers
        virtual_markers = values.get('virtual_markers').virtual_markers
        segment_connections = values.get('segment_connections')

        virtual_marker_names = set(virtual_markers.keys())

        for segment_name, segment_connection in segment_connections.items():
            if segment_connection.proximal not in markers and segment_connection.proximal not in virtual_marker_names:
                raise ValueError(f'The proximal marker {segment_connection.proximal} for {segment_name} is not in the list of markers or virtual markers.')

            if segment_connection.distal not in markers and segment_connection.distal not in virtual_marker_names:
                raise ValueError(f'The distal marker {segment_connection.distal} for {segment_name} is not in the list of markers or virtual markers.')

        return values


class Skeleton(BaseModel):
    markers:Markers
    virtual_markers: VirtualMarkers
    segments: Segments
    marker_data: Dict[str, np.ndarray] = {}  
    virtual_marker_data: Dict[str, np.ndarray] = {}
    class Config:
        arbitrary_types_allowed = True

    def integrate_freemocap_3d_data(self, freemocap_3d_data:np.ndarray):
        num_markers_in_data = freemocap_3d_data.shape[1]
        num_markers_in_model = len(self.markers.markers)
        
        if num_markers_in_data != num_markers_in_model:
            raise ValueError(
                f"The number of markers in the 3D data ({num_markers_in_data}) does not match "
                f"the number of markers in the model ({num_markers_in_model})."
            )
    
        for i, marker_name in enumerate(self.markers.markers):
            self.marker_data[marker_name] = freemocap_3d_data[:, i, :]

    def calculate_virtual_markers(self):
        # Check if actual marker data is present
        if not self.marker_data:
            raise ValueError("3d marker data must be integrated before calculating virtual markers. Run `integrate_freemocap_3d_data()` first.")

        # Iterate over the virtual markers and calculate their positions
        for vm_name, vm_info in self.virtual_markers.virtual_markers.items():
            # Initialize an array to hold the computed positions of the virtual marker
            vm_positions = np.zeros((self.marker_data[next(iter(self.marker_data))].shape[0], 3))
            for marker_name, weight in zip(vm_info['marker_names'], vm_info['marker_weights']):
                vm_positions += self.marker_data[marker_name] * weight
            self.virtual_marker_data[vm_name] = vm_positions
        
        self.marker_data.update(self.virtual_marker_data)
    
    def get_segment_markers(self, segment_name: str) -> Dict[str, np.ndarray]:
        """Returns a dictionary with the positions of the proximal and distal markers for a segment."""
        segment = self.segments.segment_connections.get(segment_name)
        if not segment:
            raise ValueError(f"Segment '{segment_name}' is not defined in the skeleton.")

        proximal_marker = self.trajectories.get(segment.proximal)
        distal_marker = self.trajectories.get(segment.distal)

        return {
            'proximal': proximal_marker,
            'distal': distal_marker
        }


    @property
    def trajectories(self):
        return self.marker_data
    
    def debug_plot(self, frame_index: int):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        """Plots the markers and segments for a given frame."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Gather all marker positions at the specified frame index
        all_marker_positions = [marker_data[frame_index] for marker_data in self.trajectories.values()]

        # Calculate the mean position along each axis
        mean_x, mean_y, mean_z = np.mean(all_marker_positions, axis=0)

        # Set the range around the mean position
        range_limit = 900
        ax.set_xlim(mean_x - range_limit, mean_x + range_limit)
        ax.set_ylim(mean_y - range_limit, mean_y + range_limit)
        ax.set_zlim(mean_z - range_limit, mean_z + range_limit)

        # Plot all markers
        for marker_name, marker_data in self.trajectories.items():
            ax.scatter(*marker_data[frame_index], label=marker_name)

        # Plot segments as lines between connected markers
        for segment_name, segment in self.segments.segment_connections.items():
            proximal_marker = self.trajectories.get(segment.proximal)
            distal_marker = self.trajectories.get(segment.distal)
            if proximal_marker is not None and distal_marker is not None:
                ax.plot(
                    [proximal_marker[frame_index, 0], distal_marker[frame_index, 0]],
                    [proximal_marker[frame_index, 1], distal_marker[frame_index, 1]],
                    [proximal_marker[frame_index, 2], distal_marker[frame_index, 2]],
                    label=f'Segment: {segment_name}'
                )

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()


markers = Markers(markers=mediapipe_body)
virtual_markers = VirtualMarkers(virtual_markers=virtual_markers)
segments = Segments(markers=markers, virtual_markers=virtual_markers, segment_connections= {name: Segment(**segment) for name, segment in segment_connections.items()})



path_to_data_folder = Path(r'D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_15_36_03_MDN_OneLeg_Trial1\output_data')
path_to_data = path_to_data_folder / 'mediapipe_body_3d_xyz.npy'

freemocap_3d_data = np.load(path_to_data)

skeleton = Skeleton(markers=markers, virtual_markers=virtual_markers, segments=segments)
skeleton.integrate_freemocap_3d_data(freemocap_3d_data)
skeleton.calculate_virtual_markers()
# skeleton.debug_plot(frame_index=300)
# trajectory = skeleton.trajectories

# left_forearm = skeleton.get_segment_markers('left_forearm')

f = 2



def calculate_segment_center_of_mass(skeleton: Skeleton, anthropometric_data: Dict[str, Dict[str, float]]) -> Dict[str, np.ndarray]:
    """
    Calculates the center of mass for each segment based on anthropometric data.
    
    Parameters:
    - skeleton: An instance of the Skeleton class containing marker trajectories.
    - anthropometric_data: A dictionary containing the segment COM length percentages.
    
    Returns:
    - A dictionary with the segment names as keys and their COM positions as values.
    """
    segment_com_data = {}

    for segment_name, segment_info in anthropometric_data.items():
        segment_coordinates = skeleton.get_segment_markers(segment_name)

        segment_proximal = segment_coordinates["proximal"]
        segment_distal = segment_coordinates["distal"]

        segment_com_length = segment_info["segment_com_length"]

        segment_com = segment_proximal + (segment_distal-segment_proximal)*segment_com_length
        segment_com_data[segment_name] = segment_com

    return segment_com_data
    f = 2

def calculate_total_body_center_of_mass(segment_center_of_mass_data: Dict[str, np.ndarray], anthropometric_data: Dict[str, Dict[str, float]]) -> np.ndarray:
    """
    Calculates the total body center of mass for each frame based on segment COM positions and anthropometric data.
    
    Parameters:
    - segment_com_data: A dictionary with segment names as keys and COM positions as values for each frame.
    - anthropometric_data: A dictionary containing segment mass percentages.
    
    Returns:
    - A numpy array containing the position of the total body center of mass for each frame.
    """
    # Assume all segments have the same number of frames
    num_frames = next(iter(segment_center_of_mass_data.values())).shape[0]
    total_body_com = np.zeros((num_frames, 3))

    for segment_name, segment_info in anthropometric_data.items():
        # Retrieve the COM position for the current segment
        segment_com = segment_center_of_mass_data.get(segment_name)
        # Retrieve the mass percentage for the current segment
        segment_mass_percentage = segment_info.get('segment_com_percentage')

        # Add the weighted segment COM to the total COM for each frame
        total_body_com += segment_com * segment_mass_percentage


    return total_body_com


segment_com_data = calculate_segment_center_of_mass(skeleton=skeleton, anthropometric_data=center_of_mass_anthropometric_data)

total_body_com_data_new = calculate_total_body_center_of_mass(segment_center_of_mass_data=segment_com_data, anthropometric_data=center_of_mass_anthropometric_data)

total_body_com_data_old = np.load(path_to_data_folder/'center_of_mass'/'total_body_center_of_mass_xyz.npy')
import matplotlib.pyplot as plt

# Plot the old total body center of mass data in red
plt.plot(total_body_com_data_old[:, 0], 'r', alpha=0.5, label='Old COM data')

# Plot the new total body center of mass data in blue
plt.plot(total_body_com_data_new[:, 0], 'b', alpha=0.5, label = 'New COM data')

# Add labels and title
plt.xlabel('frames')
plt.ylabel('Z')
plt.title('Total Body Center of Mass')
plt.legend()
# Show the plot
plt.show()



# segment_com_old = np.load(path_to_data_folder/'center_of_mass'/'segmentCOM_frame_joint_xyz.npy')
# # Select a frame index to plot
# frame_index = 300

# # Get the segment COM data for the selected frame
# segment_com_old_frame = segment_com_old[frame_index]
# segment_com_data_frame = {segment_name: segment_com[frame_index] for segment_name, segment_com in segment_com_data.items()}

# # Create a 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# mean_x, mean_y, mean_z = np.mean(segment_com_old_frame, axis=0)

# # Set the range around the mean position
# range_limit = 900
# ax.set_xlim(mean_x - range_limit, mean_x + range_limit)
# ax.set_ylim(mean_y - range_limit, mean_y + range_limit)
# ax.set_zlim(mean_z - range_limit, mean_z + range_limit)
# # Plot the segment COM data from the old format
# for segment_index, segment_com in enumerate(segment_com_old_frame):
#     ax.scatter(segment_com[0], segment_com[1], segment_com[2], color='r', alpha=0.5)

# # Plot the segment COM data from the new format
# for segment_name, segment_com in segment_com_data_frame.items():
#     ax.scatter(segment_com[0], segment_com[1], segment_com[2], color='b', alpha=0.5)

# # Set labels and title
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('Segment Center of Mass')

# # Show the plot
# plt.show()
f = 2 