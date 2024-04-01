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

mediapipe_hand = [
                "wrist",
                "thumb_cmc",
                "thumb_mcp",
                "thumb_ip",
                "thumb_tip",
                "index_finger_mcp",
                "index_finger_pip",
                "index_finger_dip",
                "index_finger_tip",
                "middle_finger_mcp",
                "middle_finger_pip",
                "middle_finger_dip",
                "middle_finger_tip",
                "ring_finger_mcp",
                "ring_finger_pip",
                "ring_finger_dip",
                "ring_finger_tip",
                "pinky_mcp",
                "pinky_pip",
                "pinky_dip",
                "pinky_tip",
                ]

segment_connections = {
"head":{
    'proximal': 'head_center',
    'distal': 'nose'
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


from pydantic import BaseModel, validator, root_validator
from typing import Dict, List

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



markers = Markers(markers=mediapipe_body)

virtual_markers = VirtualMarkers(virtual_markers=virtual_markers)


segments = Segments(markers=markers, virtual_markers=virtual_markers, segment_connections= {name: Segment(**segment) for name, segment in segment_connections.items()})

f = 2