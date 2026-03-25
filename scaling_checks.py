from pathlib import Path
import numpy as np
import pandas as pd

from skellymodels.managers.human import Human


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
PATH_TO_RECORDINGS = Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera")
TRACKER_NAME = "rtmpose_dlc"   # e.g. "rtmpose_dlc", "mediapipe_dlc", "vitpose_dlc"
REFERENCE_NAME = "qualisys"

# If True, use rigid trajectory when available; otherwise use raw xyz
PREFER_RIGID = False

# Optional: only keep segments whose names contain one of these strings
# Set to None to keep all shared segments
SEGMENT_NAME_FILTERS = None
# Example:
# SEGMENT_NAME_FILTERS = ["thigh", "shank", "foot", "pelvis"]


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def get_valid_recording_folders(path_to_recordings: Path) -> list[Path]:
    list_of_folders = [p for p in path_to_recordings.iterdir() if p.is_dir()]

    list_of_valid_folders = []
    for p in list_of_folders:
        if (p / "validation").is_dir():
            list_of_valid_folders.append(p)
        else:
            print(f"Skipping {p}")

    return list_of_valid_folders


def load_human_from_validation_folder(recording_folder: Path, tracker_name: str) -> Human | None:
    parquet_path = recording_folder / "validation" / tracker_name / "freemocap_data_by_frame.parquet"

    if not parquet_path.exists():
        print(f"Missing parquet: {parquet_path}")
        return None

    try:
        human = Human.from_parquet(parquet_path)
        return human
    except Exception as e:
        print(f"Failed to load {parquet_path}: {e}")
        return None


def get_body_trajectory(human: Human, prefer_rigid: bool = False):
    """
    Returns either human.body.rigid_xyz or human.body.xyz, depending on availability.
    """
    if prefer_rigid and getattr(human.body, "rigid_xyz", None) is not None:
        return human.body.rigid_xyz
    return human.body.xyz


def get_segment_connections(human: Human) -> dict:
    segs = human.body.anatomical_structure.segment_connections
    return segs if segs is not None else {}


def compute_segment_length_dict(human: Human, prefer_rigid: bool = False) -> dict[str, np.ndarray]:
    """
    Returns:
        {
            segment_name: array of shape (n_frames,)
        }
    """
    traj = get_body_trajectory(human, prefer_rigid=prefer_rigid)
    segment_connections = get_segment_connections(human)

    if not segment_connections:
        return {}

    segment_positions = traj.segment_data(segment_connections)

    segment_lengths = {}
    for segment_name, seg in segment_positions.items():
        proximal = seg["proximal"]
        distal = seg["distal"]

        if proximal is None or distal is None:
            continue

        lengths = np.linalg.norm(distal - proximal, axis=1)
        segment_lengths[segment_name] = lengths

    return segment_lengths


def summarize_segment_lengths(
    segment_length_dict: dict[str, np.ndarray],
    recording_name: str,
    tracker_name: str,
) -> pd.DataFrame:
    rows = []

    for segment_name, lengths in segment_length_dict.items():
        finite_lengths = lengths[np.isfinite(lengths)]

        if len(finite_lengths) == 0:
            continue

        rows.append(
            {
                "recording": recording_name,
                "tracker": tracker_name,
                "segment_name": segment_name,
                "n_frames": len(finite_lengths),
                "mean_length": float(np.mean(finite_lengths)),
                "median_length": float(np.median(finite_lengths)),
                "std_length": float(np.std(finite_lengths)),
                "min_length": float(np.min(finite_lengths)),
                "max_length": float(np.max(finite_lengths)),
            }
        )

    return pd.DataFrame(rows)


def apply_segment_filter(segment_names: list[str], filters: list[str] | None) -> list[str]:
    if filters is None:
        return segment_names

    filtered = []
    for name in segment_names:
        lower_name = name.lower()
        if any(f.lower() in lower_name for f in filters):
            filtered.append(name)
    return filtered


# ------------------------------------------------------------
# Main analysis
# ------------------------------------------------------------
list_of_valid_folders = get_valid_recording_folders(PATH_TO_RECORDINGS)

all_summary_rows = []
all_comparison_rows = []

for recording_folder in list_of_valid_folders:
    print(f"\nProcessing: {recording_folder.name}")

    ref_human = load_human_from_validation_folder(recording_folder, REFERENCE_NAME)
    tracker_human = load_human_from_validation_folder(recording_folder, TRACKER_NAME)

    if ref_human is None or tracker_human is None:
        print("  Skipping because one of the parquets could not be loaded.")
        continue

    ref_segment_lengths = compute_segment_length_dict(ref_human, prefer_rigid=PREFER_RIGID)
    tracker_segment_lengths = compute_segment_length_dict(tracker_human, prefer_rigid=PREFER_RIGID)

    if not ref_segment_lengths:
        print(f"  No segment connections found for {REFERENCE_NAME}")
        continue
    if not tracker_segment_lengths:
        print(f"  No segment connections found for {TRACKER_NAME}")
        continue

    # summarize each system separately
    ref_summary_df = summarize_segment_lengths(ref_segment_lengths, recording_folder.name, REFERENCE_NAME)
    tracker_summary_df = summarize_segment_lengths(tracker_segment_lengths, recording_folder.name, TRACKER_NAME)

    if len(ref_summary_df) > 0:
        all_summary_rows.append(ref_summary_df)
    if len(tracker_summary_df) > 0:
        all_summary_rows.append(tracker_summary_df)

    shared_segments = sorted(set(ref_segment_lengths.keys()) & set(tracker_segment_lengths.keys()))
    shared_segments = apply_segment_filter(shared_segments, SEGMENT_NAME_FILTERS)

    if len(shared_segments) == 0:
        print("  No shared segments between reference and tracker.")
        print(f"  Ref segments: {sorted(ref_segment_lengths.keys())}")
        print(f"  Tracker segments: {sorted(tracker_segment_lengths.keys())}")
        continue

    print(f"  Comparing {len(shared_segments)} shared segments")

    for segment_name in shared_segments:
        ref_lengths = ref_segment_lengths[segment_name]
        tracker_lengths = tracker_segment_lengths[segment_name]

        n = min(len(ref_lengths), len(tracker_lengths))
        ref_lengths = ref_lengths[:n]
        tracker_lengths = tracker_lengths[:n]

        valid_mask = np.isfinite(ref_lengths) & np.isfinite(tracker_lengths)
        ref_lengths = ref_lengths[valid_mask]
        tracker_lengths = tracker_lengths[valid_mask]

        if len(ref_lengths) == 0:
            continue

        ref_median = float(np.median(ref_lengths))
        tracker_median = float(np.median(tracker_lengths))
        ref_mean = float(np.mean(ref_lengths))
        tracker_mean = float(np.mean(tracker_lengths))

        ratio_median = tracker_median / ref_median if ref_median != 0 else np.nan
        ratio_mean = tracker_mean / ref_mean if ref_mean != 0 else np.nan
        abs_diff_median = tracker_median - ref_median
        pct_diff_median = (ratio_median - 1.0) * 100 if np.isfinite(ratio_median) else np.nan

        all_comparison_rows.append(
            {
                "recording": recording_folder.name,
                "segment_name": segment_name,
                "reference_tracker": REFERENCE_NAME,
                "test_tracker": TRACKER_NAME,
                "n_frames_compared": len(ref_lengths),
                "reference_median_length": ref_median,
                "tracker_median_length": tracker_median,
                "reference_mean_length": ref_mean,
                "tracker_mean_length": tracker_mean,
                "median_ratio_tracker_over_reference": ratio_median,
                "mean_ratio_tracker_over_reference": ratio_mean,
                "median_abs_diff": abs_diff_median,
                "median_pct_diff": pct_diff_median,
            }
        )

# ------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------
if len(all_summary_rows) > 0:
    summary_df = pd.concat(all_summary_rows, ignore_index=True)
else:
    summary_df = pd.DataFrame()

if len(all_comparison_rows) > 0:
    comparison_df = pd.DataFrame(all_comparison_rows)
else:
    comparison_df = pd.DataFrame()

output_folder = PATH_TO_RECORDINGS / "segment_length_diagnostics"
output_folder.mkdir(exist_ok=True, parents=True)

summary_path = output_folder / f"{TRACKER_NAME}_and_{REFERENCE_NAME}_segment_length_summary.csv"
comparison_path = output_folder / f"{TRACKER_NAME}_vs_{REFERENCE_NAME}_segment_length_comparison.csv"
grouped_path = output_folder / f"{TRACKER_NAME}_vs_{REFERENCE_NAME}_segment_length_grouped_summary.csv"

summary_df.to_csv(summary_path, index=False)
comparison_df.to_csv(comparison_path, index=False)

if len(comparison_df) > 0:
    grouped_df = (
        comparison_df
        .groupby("segment_name", as_index=False)
        .agg(
            n_recordings=("recording", "nunique"),
            median_of_median_ratios=("median_ratio_tracker_over_reference", "median"),
            mean_of_median_ratios=("median_ratio_tracker_over_reference", "mean"),
            std_of_median_ratios=("median_ratio_tracker_over_reference", "std"),
            min_of_median_ratios=("median_ratio_tracker_over_reference", "min"),
            max_of_median_ratios=("median_ratio_tracker_over_reference", "max"),
            median_pct_diff=("median_pct_diff", "median"),
            mean_pct_diff=("median_pct_diff", "mean"),
        )
        .sort_values("mean_of_median_ratios")
    )
    grouped_df.to_csv(grouped_path, index=False)

    print("\n--- Grouped summary across recordings ---")
    print(grouped_df.to_string(index=False))
else:
    grouped_df = pd.DataFrame()
    print("\nNo comparison rows were generated.")

print(f"\nSaved summary to: {summary_path}")
print(f"Saved per-recording comparison to: {comparison_path}")
if len(grouped_df) > 0:
    print(f"Saved grouped summary to: {grouped_path}")