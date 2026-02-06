from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from skellymodels.managers.human import Human

recording_root = Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera")

trackers = ["rtmpose", "mediapipe_dlc", "qualisys"]

# IMPORTANT: make z a numpy array (float) and unit-normalize it
z = np.array([0.0, 0.0, 1.0], dtype=float)
z /= np.linalg.norm(z)

recordings_list = [
    recording_root / "sesh_2023-06-07_12_38_16_TF01_leg_length_neg_5_trial_1",
    recording_root / "sesh_2023-06-07_12_43_15_TF01_leg_length_neg_25_trial_1",
    recording_root / "sesh_2023-06-07_12_46_54_TF01_leg_length_neutral_trial_1",
    recording_root / "sesh_2023-06-07_12_50_56_TF01_leg_length_pos_25_trial_1",
    recording_root / "sesh_2023-06-07_12_55_21_TF01_leg_length_pos_5_trial_1",
]

mapping = {
    "sesh_2023-06-07_12_38_16_TF01_leg_length_neg_5_trial_1": -12.5,
    "sesh_2023-06-07_12_43_15_TF01_leg_length_neg_25_trial_1": -6.25,
    "sesh_2023-06-07_12_46_54_TF01_leg_length_neutral_trial_1": "neutral",
    "sesh_2023-06-07_12_50_56_TF01_leg_length_pos_25_trial_1": 6.25,
    "sesh_2023-06-07_12_55_21_TF01_leg_length_pos_5_trial_1": 12.5,
}


def compute_pelvic_obliquity(recording: Path, tracker: str, z_axis: np.ndarray) -> np.ndarray:
    path_to_data = recording / "validation" / tracker
    if not path_to_data.exists():
        raise FileNotFoundError(path_to_data)

    human: Human = Human.from_data(path_to_data)

    left_pelvis = human.body.xyz.as_dict["left_hip"]
    right_pelvis = human.body.xyz.as_dict["right_hip"]

    pelvis_vector = right_pelvis - left_pelvis

    dz = pelvis_vector @ z_axis
    pelvis_horizontal = pelvis_vector - dz[:, None] * z_axis
    dh = np.linalg.norm(pelvis_horizontal, axis=1)

    eps = 1e-9
    return np.degrees(np.arctan2(dz, np.maximum(dh, eps)))


# -----------------------
# SUBPLOT FIGURE
# -----------------------
fig = make_subplots(
    rows=len(recordings_list),
    cols=1,
    shared_xaxes=False,
    shared_yaxes=True,
    vertical_spacing=0.05,
    subplot_titles=[
        f"Leg length condition: {mapping[r.stem]}"
        for r in recordings_list
    ],
)

for row, recording in enumerate(recordings_list, start=1):
    for tracker in trackers:
        try:
            y = compute_pelvic_obliquity(recording, tracker, z)
        except Exception as e:
            print(f"[SKIP] {recording.stem} | {tracker}: {e}")
            continue

        fig.add_trace(
            go.Scatter(
                x=np.arange(len(y)),
                y=y,
                mode="lines",
                name=tracker,
                line=dict(width=2),
                showlegend=(row == 1),  # legend only once
            ),
            row=row,
            col=1,
        )

fig.update_layout(
    title="Pelvic obliquity by tracker across leg-length conditions",
    xaxis_title="Frame",
    yaxis_title="Pelvic obliquity (deg)",
    template="plotly_white",
    height=300 * len(recordings_list),
    legend_title="Tracker",
)

fig.show()
