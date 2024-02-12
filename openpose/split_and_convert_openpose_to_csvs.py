import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union

def save_openpose_parts_to_csvs(data_array, output_directory):
    # OpenPose BODY_25 body part names
    body_part_names = [
        "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
        "left_shoulder", "left_elbow", "left_wrist", "mid_hip", "right_hip",
        "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle",
        "right_eye", "left_eye", "right_ear", "left_ear", "left_big_toe",
        "left_small_toe", "left_heel", "right_big_toe", "right_small_toe", "right_heel"
    ]

    # Define marker counts for each part
    body_markers = len(body_part_names)
    hand_markers = 21  # Each hand has the same number of markers
    face_markers = 70
    total_markers = body_markers + 2 * hand_markers + face_markers

    # Split data into parts
    body_data = data_array[:, :body_markers, :]
    left_hand_data = data_array[:, body_markers:body_markers + hand_markers, :]
    right_hand_data = data_array[:, body_markers + hand_markers:body_markers + 2 * hand_markers, :]
    face_data = data_array[:, body_markers + 2 * hand_markers:total_markers, :]

    # Prepare output directory
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    # Save function for convenience
    def save_to_csv(data, part_name, output_dir, part_names=None):
        # Flatten the data across markers and dimensions for CSV format
        flattened_data = data.reshape(data.shape[0], -1)
        
        if part_names:
            # If part names are provided, generate columns with detailed names
            columns = [f"{name}_{dim}" for name in part_names for dim in ['x', 'y', 'z']]
        else:
            # Generate generic column names based on the number of markers, corrected to not multiply by 3
            columns = [f"{part_name}_{i}_{dim}" for i in range(data.shape[1]) for dim in ['x', 'y', 'z']]

        assert flattened_data.shape[1] == len(columns), f"Mismatch in flattened data columns ({flattened_data.shape[1]}) and generated column names ({len(columns)})"
        

        # Create a DataFrame and save to CSV
        df = pd.DataFrame(flattened_data, columns=columns)
        csv_path = output_dir / f"openpose_{part_name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved {part_name} data to {csv_path}")

    # Save each part to CSV
    save_to_csv(body_data, "body", output_directory, body_part_names)
    save_to_csv(left_hand_data, "left_hand", output_directory)
    save_to_csv(right_hand_data, "right_hand", output_directory)
    save_to_csv(face_data, "face", output_directory)


# Example usage
data_array = np.load(Path(r'D:\steen_pantsOn_gait_3_cameras\output_data\openpose_postprocessed_3d_xyz.npy'))
output_directory = Path(r"D:\steen_pantsOn_gait_3_cameras\output_data\openpose_data")
save_openpose_parts_to_csvs(data_array, output_directory)
