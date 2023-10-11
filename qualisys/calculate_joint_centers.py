from rich.progress import track
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

