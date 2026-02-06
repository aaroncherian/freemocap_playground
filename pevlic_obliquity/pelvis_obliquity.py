from pathlib import Path
from skellymodels.managers.human import Human
import numpy as np
import plotly.graph_objects as go

recording_root = Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera")
tracker = "rtmpose"

# IMPORTANT: make z a numpy array (float) and unit-normalize it
z = np.array([0.0, 0.0, 1.0])
z = z / np.linalg.norm(z)

recordings_list = [
    recording_root/"sesh_2023-06-07_12_38_16_TF01_leg_length_neg_5_trial_1",
    recording_root/"sesh_2023-06-07_12_43_15_TF01_leg_length_neg_25_trial_1",
    recording_root/"sesh_2023-06-07_12_46_54_TF01_leg_length_neutral_trial_1",
    recording_root/"sesh_2023-06-07_12_50_56_TF01_leg_length_pos_25_trial_1",
    recording_root/"sesh_2023-06-07_12_55_21_TF01_leg_length_pos_5_trial_1",
]

mapping = {
    "sesh_2023-06-07_12_38_16_TF01_leg_length_neg_5_trial_1": -12.5,
    "sesh_2023-06-07_12_43_15_TF01_leg_length_neg_25_trial_1": -6.25,
    "sesh_2023-06-07_12_46_54_TF01_leg_length_neutral_trial_1": "neutral",
    "sesh_2023-06-07_12_50_56_TF01_leg_length_pos_25_trial_1": 6.25,
    "sesh_2023-06-07_12_55_21_TF01_leg_length_pos_5_trial_1": 12.5,
}

dict_pelvic_obliquity = {}

for recording in recordings_list:
    path_to_data = recording / "validation" / tracker
    human: Human = Human.from_data(path_to_data)

    left_pelvis = human.body.xyz.as_dict["left_hip"]   # shape (N,3)
    right_pelvis = human.body.xyz.as_dict["right_hip"] # shape (N,3)

    pelvis_vector = right_pelvis - left_pelvis         # (N,3)

    # vertical component along z (per frame): (N,)
    dz = pelvis_vector @ z

    # horizontal component (remove vertical component): (N,3)
    pelvis_horizontal = pelvis_vector - dz[:, None] * z

    # horizontal magnitude per frame: (N,)
    dh = np.linalg.norm(pelvis_horizontal, axis=1)

    # avoid divide-by-zero weirdness (shouldn't happen, but safe)
    eps = 1e-9
    pelvic_obliquity = np.degrees(np.arctan2(dz, np.maximum(dh, eps)))  # (N,)

    label = str(mapping[recording.stem])
    dict_pelvic_obliquity[label] = pelvic_obliquity


# -----------------------
# PLOTLY OVERLAY PLOT
# -----------------------
fig = go.Figure()

# Optional: sort legend order nicely (numbers then neutral)
def sort_key(k):
    if k == "neutral":
        return (1, 0)
    return (0, float(k))

for label in sorted(dict_pelvic_obliquity.keys(), key=sort_key):
    y = dict_pelvic_obliquity[label]
    x = np.arange(len(y))  # frame index

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=label,
            line=dict(width=2),
            opacity=0.9,
        )
    )

fig.update_layout(
    title="Pelvic obliquity (yaw-invariant) overlay across conditions",
    xaxis_title="Frame",
    yaxis_title="Pelvic obliquity (deg)",
    legend_title="Condition",
    template="plotly_white",
    height=550,
)

fig.show()
