from models.skeleton import Skeleton
from models.joint_hierarchy import JointHierarchy
from typing import Dict, List

from .enforce_rigid_bones import enforce_rigid_bones
from .calculate_bone_statistics import calculate_bone_lengths_and_statistics

def enforce_rigid_bones_from_skeleton(skeleton:Skeleton):

    bone_lengths_and_statistcs = calculate_bone_lengths_and_statistics(
        marker_data=skeleton.marker_data, 
        segment_connections=skeleton.segments
    )


    rigid_marker_data = enforce_rigid_bones(
        marker_data=skeleton.marker_data, 
        segment_connections=skeleton.segments, 
        bone_lengths_and_statistics=bone_lengths_and_statistcs, 
        joint_hierarchy=skeleton.joint_hierarchy
    )

    return rigid_marker_data