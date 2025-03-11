from skellymodels.experimental.model_redo.managers.human import Human
from skellymodels.experimental.model_redo.tracker_info.model_info import ModelInfo
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



# Load the data
path_to_recording = Path(r'D:\2025_03_05_atc_testing\freemocap_data\2025-03-05T16_06_17_gmt-5_atc_trial_3')
# path_to_output_data = path_to_recording / 'saved_data' / 'npy' / 'body_frame_name_xyz.npy'
path_to_output_data = path_to_recording / 'output_data'/ 'mediapipe_rigid_bones_3d.npy'


model_info = ModelInfo(config_path=Path(__file__).parent.parent/'tracker_info'/'mediapipe_just_body.yaml')
human = Human.from_numpy_array(
    name='human',
    model_info=model_info,
    tracked_points_numpy_array=np.load(path_to_output_data)
)

# Get segment_data directly as used in the original code
segment_data = human.body.trajectories['3d_xyz'].segment_data

# Function to check rigidity of all segments
def check_all_rigid_segments(segment_data, tolerance=0.001):
    """
    Checks whether all rigid body segments maintain constant length over time.
    
    Parameters:
        segment_data (dict): Dictionary where each key is a segment name, 
                             and each value contains 'proximal' and 'distal' arrays.
        tolerance (float): Allowed variation in segment length (default 1mm, or 0.001m).
    
    Returns:
        dict: A dictionary containing analysis results for each segment.
    """
    results = {}

    for segment_name, segment in segment_data.items():
        proximal = np.array(segment['proximal'])  # Shape: (n_frames, 3)
        distal = np.array(segment['distal'])      # Shape: (n_frames, 3)

        if proximal.shape != distal.shape:
            raise ValueError(f"Mismatch in proximal/distal shape for segment {segment_name}")

        # Compute Euclidean distance for each frame
        segment_lengths = np.linalg.norm(proximal - distal, axis=1)
        
        # Get both reference methods
        expected_length = segment_lengths[0]  # First frame as reference (original method)
        mean_length = np.nanmean(segment_lengths)
        
        # Compute statistics
        deviations = np.abs(segment_lengths - expected_length)
        max_change = np.max(deviations)
        std_deviation = np.std(segment_lengths)
        cv = (std_deviation / mean_length) * 100  # Coefficient of variation in percent
        absolute_range = np.max(segment_lengths) - np.min(segment_lengths)

        # Check if any deviation exceeds the tolerance
        violates_rigidity = np.any(deviations > tolerance)
        violates_cv = cv > 0.5  # Using 0.5% CV as threshold

        results[segment_name] = {
            "lengths": segment_lengths,
            "expected_length": expected_length,
            "mean_length": mean_length,
            "max_change": max_change,
            "std_deviation": std_deviation,
            "cv": cv,
            "absolute_range": absolute_range,
            "violates_rigidity": violates_rigidity,
            "violates_cv": violates_cv
        }

    return results

# Run the rigidity check
results = check_all_rigid_segments(segment_data)

# Sort segments by maximum change (most problematic first)
sorted_segments = sorted(results.items(), key=lambda x: x[1]['max_change'], reverse=True)

# Create visualization
plt.figure(figsize=(15, 12))
plt.suptitle("Bone Length Stability Across Frames", fontsize=16)

# Plot 1: Top 5 most variable segments + spine
plt.subplot(2, 1, 1)
# Always include the spine if it exists
spine_included = False
for i, (segment_name, segment_results) in enumerate(sorted_segments[:5]):
    plt.plot(segment_results['lengths'], label=f"{segment_name} (max dev: {segment_results['max_change']:.4f})")
    if segment_name == 'spine':
        spine_included = True

# Add spine if it wasn't in the top 5
if not spine_included and 'spine' in results:
    spine_results = results['spine']
    plt.plot(spine_results['lengths'], 'k--', linewidth=2, 
             label=f"spine (max dev: {spine_results['max_change']:.4f})")

plt.title("Segments with Largest Deviations from First Frame")
plt.xlabel("Frame")
plt.ylabel("Length (units)")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Maximum change for all segments
plt.subplot(2, 1, 2)
segment_names = [s[0] for s in sorted_segments]
max_changes = [s[1]['max_change'] for s in sorted_segments]

bars = plt.bar(segment_names, max_changes)
plt.xticks(rotation=90)
plt.title("Maximum Deviation from First Frame for Each Segment")
plt.ylabel("Maximum Deviation (units)")
plt.axhline(y=0.001, color='r', linestyle='--', label="1mm Threshold")
plt.legend()
plt.grid(True, axis='y', alpha=0.3)

# Highlight bars based on max_change value
# Find the spine index if it exists
spine_index = -1
for i, name in enumerate(segment_names):
    if name == 'spine':
        spine_index = i
        break

# Highlight the bars
for i, max_change in enumerate(max_changes):
    if i == spine_index:
        # Highlight spine in a unique color (blue)
        bars[i].set_color('royalblue') 
        bars[i].set_edgecolor('navy')
        bars[i].set_linewidth(2)
    elif max_change > 0.001:
        bars[i].set_color('r')
    else:
        bars[i].set_color('g')

plt.tight_layout(rect=[0, 0, 1, 0.95])

# Create a summary table of statistics
plt.figure(figsize=(12, 8))
plt.axis('off')
plt.title("Segment Length Statistics", fontsize=16)

# Create table data
table_data = []
headers = ["Segment", "First Frame", "Mean Length", "Max Dev", "CV (%)", "Rigid 1mm?", "Rigid CV?"]

for segment_name, segment_results in sorted_segments:
    is_rigid_abs = "Yes" if segment_results['max_change'] < 0.001 else "No"
    is_rigid_cv = "Yes" if segment_results['cv'] < 0.5 else "No"
    
    table_data.append([
        segment_name, 
        f"{segment_results['expected_length']:.4f}", 
        f"{segment_results['mean_length']:.4f}", 
        f"{segment_results['max_change']:.4f}",
        f"{segment_results['cv']:.2f}", 
        is_rigid_abs,
        is_rigid_cv
    ])

# Create table
cell_colors = []
for row in table_data:
    is_rigid_abs = row[5]
    is_rigid_cv = row[6]
    
    if is_rigid_abs == "Yes" and is_rigid_cv == "Yes":
        # Both methods say it's rigid - green
        cell_colors.append(["w", "w", "w", "w", "w", "lightgreen", "lightgreen"])
    elif is_rigid_abs == "No" and is_rigid_cv == "No":
        # Both methods say it's not rigid - red
        cell_colors.append(["w", "w", "w", "w", "w", "lightcoral", "lightcoral"])
    else:
        # Methods disagree - yellow highlight where rigidity is claimed
        colors = ["w", "w", "w", "w", "w", "w", "w"]
        colors[5] = "lightgreen" if is_rigid_abs == "Yes" else "lightcoral"
        colors[6] = "lightgreen" if is_rigid_cv == "Yes" else "lightcoral"
        cell_colors.append(colors)

plt.table(
    cellText=table_data,
    colLabels=headers,
    loc='center',
    cellColours=cell_colors,
    cellLoc='center'
)

# Add a specific test for spine only
if 'spine' in results:
    plt.figure(figsize=(12, 8))
    spine_results = results['spine']
    
    # Plot spine length over time
    plt.subplot(3, 1, 1)
    plt.plot(spine_results['lengths'], 'b-', linewidth=2)
    plt.axhline(y=spine_results['mean_length'], color='r', linestyle='--', 
               label=f"Mean: {spine_results['mean_length']:.4f}")
    plt.axhline(y=spine_results['expected_length'], color='g', linestyle=':', 
               label=f"First frame: {spine_results['expected_length']:.4f}")
    plt.fill_between(range(len(spine_results['lengths'])), 
                    spine_results['mean_length'] - spine_results['std_deviation'], 
                    spine_results['mean_length'] + spine_results['std_deviation'], 
                    color='r', alpha=0.2, 
                    label=f"±1 Std: {spine_results['std_deviation']:.4f}")
    plt.title("Spine Length Variation Over Time")
    plt.xlabel("Frame")
    plt.ylabel("Length (units)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot deviation from first frame
    plt.subplot(3, 1, 2)
    deviations = np.abs(spine_results['lengths'] - spine_results['expected_length'])
    plt.plot(deviations, 'r-', linewidth=2)
    plt.axhline(y=0.001, color='k', linestyle='--', 
               label="1mm threshold")
    plt.title("Absolute Deviation from First Frame (Original Method)")
    plt.xlabel("Frame")
    plt.ylabel("Deviation (units)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot frame-to-frame changes
    plt.subplot(3, 1, 3)
    frame_diffs = np.abs(spine_results['lengths'][1:] - spine_results['lengths'][:-1])
    plt.plot(range(1, len(spine_results['lengths'])), frame_diffs, 'g-')
    plt.axhline(y=np.mean(frame_diffs), color='r', linestyle='--', 
               label=f"Mean change: {np.mean(frame_diffs):.6f}")
    plt.title("Frame-to-Frame Changes in Spine Length")
    plt.xlabel("Frame")
    plt.ylabel("Absolute Change")
    plt.yscale('log')  # Using log scale to better visualize small changes
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()

# Print summary of test results
total_segments = len(segment_data)
rigid_segments_abs = sum(1 for _, results in results.items() if not results['violates_rigidity'])
rigid_segments_cv = sum(1 for _, results in results.items() if not results['violates_cv'])

print(f"\nRigid Bones Test Results:")
print(f"- Total segments analyzed: {total_segments}")
print(f"- Segments passing 1mm test: {rigid_segments_abs} ({rigid_segments_abs/total_segments*100:.1f}%)")
print(f"- Segments passing 0.5% CV test: {rigid_segments_cv} ({rigid_segments_cv/total_segments*100:.1f}%)")

# If any segments failed, print their details
segments_failing_abs = [s for s, r in results.items() if r['violates_rigidity']]

if segments_failing_abs:
    print("\nSegments failing 1mm test (original method):")
    for segment_name in segments_failing_abs:
        r = results[segment_name]
        print(f"- {segment_name}:")
        print(f"  First frame length: {r['expected_length']:.4f}")
        print(f"  Mean length: {r['mean_length']:.4f}")
        print(f"  Max deviation: {r['max_change']:.4f} (Threshold: 0.001)")
        print(f"  CV: {r['cv']:.2f}%")

# Special focus on the spine
if 'spine' in results:
    spine_results = results['spine']
    print("\nSPINE ANALYSIS:")
    print(f"- Length in first frame: {spine_results['expected_length']:.4f}")
    print(f"- Mean length: {spine_results['mean_length']:.4f}")
    print(f"- Maximum deviation from first frame: {spine_results['max_change']:.4f}")
    print(f"- Standard deviation: {spine_results['std_deviation']:.4f}")
    print(f"- Coefficient of variation: {spine_results['cv']:.2f}%")
    print(f"- Absolute range (max-min): {spine_results['absolute_range']:.4f}")
    print(f"- Is rigid by 1mm threshold: {'Yes' if not spine_results['violates_rigidity'] else 'No'}")
    print(f"- Is rigid by 0.5% CV threshold: {'Yes' if not spine_results['violates_cv'] else 'No'}")

plt.tight_layout()
plt.show()