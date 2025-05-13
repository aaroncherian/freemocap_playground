import tomllib
import numpy as np
from itertools import combinations
import math

def load_camera_data(file_path):
    """Load camera data from a TOML file."""
    with open(file_path, "rb") as f:
        data = tomllib.load(f)
    
    # Extract camera positions (translations)
    cameras = {}
    for key, value in data.items():
        if key.startswith("cam_"):
            cameras[key] = {
                "name": value["name"],
                "translation": np.array(value["translation"])
            }
    
    return cameras

def calculate_distances(cameras):
    """Calculate distances between all pairs of cameras."""
    distances = {}
    for (cam1_id, cam1), (cam2_id, cam2) in combinations(cameras.items(), 2):
        distance = np.linalg.norm(cam1["translation"] - cam2["translation"])
        distances[(cam1_id, cam2_id)] = {
            "camera_names": (cam1["name"], cam2["name"]),
            "distance": distance
        }
    
    return distances

def main():
    # Load camera data from both files
    aligned_calibration_path = r"C:\Users\jonma\Sync\freemocap-stuff\freemocap-clients\ben-scholl\data\2025-04-28\2025-04-28-calibration\2025-04-28-calibration_camera_calibration_aligned.toml"
    original_calibration_path = r"C:\Users\jonma\Sync\freemocap-stuff\freemocap-clients\ben-scholl\data\2025-04-28\2025-04-28-calibration\2025-04-28-calibration_camera_calibration.toml"


    original_data = load_camera_data(original_calibration_path)
    aligned_data = load_camera_data(aligned_calibration_path)
    
    # Calculate distances for both datasets
    original_distances = calculate_distances(original_data)
    aligned_distances = calculate_distances(aligned_data)
    
    # Print results
    print("Camera Distances (in mm)")
    print("=" * 80)
    print(f"{'Camera Pair':<15} {'Original Distance':>20} {'Aligned Distance':>20} {'Difference':>15}")
    print("-" * 80)
    
    for (cam1_id, cam2_id), original in original_distances.items():
        aligned = aligned_distances[(cam1_id, cam2_id)]
        diff = abs(original["distance"] - aligned["distance"])
        
        print(f"{cam1_id}-{cam2_id:<10} {original['distance']:>20.2f} {aligned['distance']:>20.2f} {diff:>15.2f}")
    
    # Calculate average distances
    original_avg = sum(d["distance"] for d in original_distances.values()) / len(original_distances)
    aligned_avg = sum(d["distance"] for d in aligned_distances.values()) / len(aligned_distances)
    
    print("-" * 80)
    print(f"Average:      {original_avg:>20.2f} {aligned_avg:>20.2f}")

if __name__ == "__main__":
    main()