from pathlib import Path
import toml
import itertools
import numpy as np
import pandas as pd
import cv2  # OpenCV for Rodrigues transformation

def calculate_rotation_angles_from_combination(rotation_combinations: list[tuple]) -> list[float]:
    """
    Given a list of rotation matrix combinations, computes the relative rotation angles.
    """
    angles = []
    for combo in rotation_combinations:
        angles.append(calculate_rotation_angle(combo))
    return angles

def calculate_rotation_angle(rotation_pair: tuple) -> float:
    """
    Computes the relative rotation angle (in degrees) between two cameras.
    """
    R1, _ = cv2.Rodrigues(np.array(rotation_pair[0]))  # Convert Rodrigues to 3x3 rotation matrix
    R2, _ = cv2.Rodrigues(np.array(rotation_pair[1]))

    R_rel = R2 @ R1.T  # Compute relative rotation matrix
    trace_value = np.trace(R_rel)
    theta = np.arccos((trace_value - 1) / 2)  # Compute rotation angle in radians
    return np.degrees(theta)  # Convert to degrees

def get_calibration_data(path_to_folder_of_tomls: Path):
    """
    Loads calibration data from all TOML files in the specified folder.
    """
    return {
        calibration_data['metadata']['system']: calibration_data
        for file in path_to_folder_of_tomls.glob('*.toml')
        if (calibration_data := toml.load(file))
    }

def extract_rotation_data(calibration_dict: dict, num_cams: int):
    """
    Extracts rotation vectors from the calibration dictionary.
    """
    return {
        system_name: [calibration_data[f'cam_{i}']['rotation'] for i in range(num_cams)]
        for system_name, calibration_data in calibration_dict.items()
    }

def compute_pairwise_rotation_angles(rotations_data: dict):
    """
    Computes pairwise relative rotation angles for all systems.
    """
    rotation_angles = {}
    for system, rotations in rotations_data.items():
        combinations = list(itertools.combinations(rotations, 2))
        rotation_angles[system] = np.sort(calculate_rotation_angles_from_combination(combinations))

    return rotation_angles

def run_pairwise_rotation_calculation(path_to_folder_of_tomls: Path, num_cams: int) -> pd.DataFrame:
    """
    Main function to compute pairwise rotation angles between cameras for each OS.
    """
    calibration_dict = get_calibration_data(path_to_folder_of_tomls)
    rotations_dict = extract_rotation_data(calibration_dict=calibration_dict, num_cams=num_cams)
    rotation_angles_dict = compute_pairwise_rotation_angles(rotations_data=rotations_dict)
    return pd.DataFrame.from_dict(rotation_angles_dict)

if __name__ == '__main__':
    path_to_folder_of_tomls = Path(r"D:\system_testing\super_organized_folder\numerically_sorted_videos\calibrations")
    df = run_pairwise_rotation_calculation(path_to_folder_of_tomls=path_to_folder_of_tomls, num_cams=3)
    print(df)