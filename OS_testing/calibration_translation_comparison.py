from pathlib import Path
import toml
import itertools
import numpy as np
import pandas as pd

def calculate_distances_from_combination(distance_combination:list[tuple]) -> list[float]:
    distances = []
    for combo in distance_combination:
        distances.append(calculate_distance(combo))
    return distances


def calculate_distance(point_list:list) -> float:
    return np.linalg.norm(np.array(point_list[0]) - np.array(point_list[1]))

def get_calibration_data(path_to_folder_of_tomls:Path):
    return {calibration_data['metadata']['system']: calibration_data for file in path_to_folder_of_tomls.glob('*.toml') if (calibration_data := toml.load(file))}

def extract_translation_data(calibration_dict:dict, num_cams:int):
    return {system_name: [calibration_data[f'cam_{i}']['translation'] for i in range(num_cams)] for system_name, calibration_data in calibration_dict.items()}

def compute_pairwise_distances(translations_data:dict):
    distances = {}
    for system, translations in translations_data.items():
        combinations = list(itertools.combinations(translations,2))
        distances[system] = np.sort(calculate_distances_from_combination(combinations))
    
    return distances

def run_pairwise_distance_calculation(path_to_folder_of_tomls:Path, num_cams:int)->pd.DataFrame:
    calibration_dict = get_calibration_data(path_to_folder_of_tomls)
    translations_dict = extract_translation_data(calibration_dict=calibration_dict, num_cams = num_cams)
    distances_dict = compute_pairwise_distances(translations_data=translations_dict)
    return pd.DataFrame.from_dict(distances_dict)

if __name__ == '__main__':

    path_to_folder_of_tomls = Path(r"D:\system_testing\calibrations\calibration_no_pin")
    df = run_pairwise_distance_calculation(path_to_folder_of_tomls=path_to_folder_of_tomls,
                                           num_cams=3)
    print(df)
f = 2

