from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


TRACKERS = ["rtmpose", "vitpose", "mediapipe"]

tracker_colors = {
    "mediapipe": "#1f77b4",   # blue
    "rtmpose": "#2ca02c", # green
    "vitpose": "#d62728",   # red
}

path_to_recordings_list = [
    Path(r"D:\validation\data\2025_07_31_JSM_pilot\freemocap"),
    Path(r"D:\validation\data\2025_09_03_OKK\freemocap"),
    Path(r"D:\validation\data\2025-11-04_ATC"),
    Path(r"D:\validation\data\2026_01_26_KK"),
    Path(r"D:\validation\data\2026-01-30-JTM"),
    Path(r"D:\validation\data\2026_03_04_ML"),
]


def get_participant_label(path: Path) -> str:
    """
    Handles cases like:
    - D:/.../2025_07_31_JSM_pilot/freemocap -> JSM
    - D:/.../2025-11-04_ATC -> ATC
    """
    if path.name.lower() == "freemocap":
        folder_name = path.parent.name
    else:
        folder_name = path.name

    parts = folder_name.replace("-", "_").split("_")

    # usually something like YYYY_MM_DD_CODE or YYYY-MM-DD-CODE
    # grab the first all-caps-ish token after the date
    for part in parts[3:]:
        if part:
            return part

    return folder_name


def collect_scaling_data(path_to_recordings_list, trackers):
    rows = []

    for tracker in trackers:
        for starting_path in path_to_recordings_list:
            print(f"\nChecking {starting_path} for tracker={tracker}")

            if not starting_path.exists():
                print(f"  Path does not exist, skipping.")
                continue

            participant = get_participant_label(starting_path)

            folders = [p for p in starting_path.iterdir() if p.is_dir()]
            valid_folders = []

            for folder in folders:
                if (folder / "validation").is_dir():
                    valid_folders.append(folder)
                else:
                    print(f"  Skipping {folder}")

            for valid_folder in valid_folders:
                path_to_scaling = valid_folder / "validation" / tracker / "transformation_3d.npy"

                if not path_to_scaling.exists():
                    print(f"  Missing scaling file: {path_to_scaling}")
                    continue

                scaling = float(np.load(path_to_scaling)[-1])

                rows.append(
                    {
                        "tracker": tracker,
                        "participant": participant,
                        "recording_root": str(starting_path),
                        "trial_folder": valid_folder.name,
                        "scaling_factor": scaling,
                    }
                )

    return pd.DataFrame(rows)


df = collect_scaling_data(path_to_recordings_list, TRACKERS)

print("\nRaw data:")
print(df)

participant_summary = (
    df.groupby(["tracker", "participant"], as_index=False)["scaling_factor"]
    .agg(["count", "mean", "median", "std", "min", "max"])
    .reset_index()
)

overall_summary = (
    df.groupby("tracker", as_index=False)["scaling_factor"]
    .agg(["count", "mean", "median", "std", "min", "max"])
    .reset_index()
)

print("\nParticipant summary:")
print(participant_summary)

print("\nOverall summary:")
print(overall_summary)


participant_order = sorted(df["participant"].unique())
tracker_order = ["rtmpose", "mediapipe", "vitpose"]  # put vitpose last since it tends to differ more

tracker_display_names = {
    "rtmpose": "RTMPose",
    "mediapipe": "MediaPipe",
    "vitpose": "ViTPose",
}

tracker_symbols = {
    "rtmpose": "circle",
    "mediapipe": "square",
    "vitpose": "diamond",
}

fig = make_subplots(
    rows=2,
    cols=1,
    shared_yaxes=True,
    vertical_spacing=0.12,
    row_heights=[0.65, 0.35],
    subplot_titles=(
        "Participant-level scaling factors",
        "Overall scaling factor distribution by tracker",
    ),
)

# ---- Top panel: participant-level trial points + participant means ----
for tracker in tracker_order:
    tracker_df = df[df["tracker"] == tracker].copy()

    fig.add_trace(
        go.Box(
            x=tracker_df["participant"],
            y=tracker_df["scaling_factor"],
            name=tracker_display_names[tracker],
            boxpoints="all",
            pointpos=0,
            jitter=0.25,
            marker_symbol=tracker_symbols[tracker],
            marker_color=tracker_colors[tracker],
            line_color=tracker_colors[tracker],
            opacity=0.6,
            legendgroup=tracker,
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    tracker_means = (
        tracker_df.groupby("participant", as_index=False)["scaling_factor"]
        .mean()
        .rename(columns={"scaling_factor": "participant_mean"})
    )


# ---- Bottom panel: overall distributions ----
for tracker in tracker_order:
    tracker_df = df[df["tracker"] == tracker].copy()

    fig.add_trace(
        go.Box(
            x=[tracker_display_names[tracker]] * len(tracker_df),
            y=tracker_df["scaling_factor"],
            name=tracker_display_names[tracker],
            boxpoints="all",
            pointpos=0,
            jitter=0.2,
            marker_symbol=tracker_symbols[tracker],
            opacity=0.75,
            legendgroup=f"overall_{tracker}",
            marker_color=tracker_colors[tracker],
            line_color=tracker_colors[tracker],
            showlegend=False,
        ),
        row=2,
        col=1,
    )

# reference line at 1.0
fig.add_hline(y=1.0, line_dash="dash", row=1, col=1)
fig.add_hline(y=1.0, line_dash="dash", row=2, col=1)

fig.update_xaxes(
    categoryorder="array",
    categoryarray=participant_order,
    row=1,
    col=1,
)
fig.update_yaxes(title_text="Scaling factor", row=1, col=1)
fig.update_yaxes(title_text="Scaling factor", row=2, col=1)
fig.update_xaxes(title_text="Participant", row=1, col=1)
fig.update_xaxes(title_text="Tracker", row=2, col=1)

fig.update_layout(
    title="Scaling factor comparison across trackers",
    height=900,
    width=1100,
    template="plotly_white",
)

fig.show()

df = collect_scaling_data(path_to_recordings_list, TRACKERS)
output_path = Path(r"D:\validation\scaling_factors_by_trial_participant.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output_path, index=False)

print(f"Saved raw scaling values to: {output_path}")