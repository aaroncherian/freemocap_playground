from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# -----------------------
# USER CONFIG
# -----------------------
recording_root = Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera")

trackers = ["rtmpose", "mediapipe_dlc", "qualisys"]

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

# Your palette (keys reused; we'll just assign in order to the 5 trials)
CONDITION_STYLE = {
    "neg_5_6": {"line": "#94342b"},
    "neg_2_8": {"line": "#d39182"},
    "neutral": {"line": "#524F4F"},
    "pos_2_8": {"line": "#7bb6c6"},
    "pos_5_6": {"line": "#447c8e"},
}

# Assign these five colors (in this order) to the five leg-length sessions.
# If you want a different mapping, just reorder this list.
COLOR_ORDER = ["neg_5_6", "neg_2_8", "neutral", "pos_2_8", "pos_5_6"]

METRIC = "stance_pct"
SIDE = "left"

# Histogram settings
NBINS = 30
HISTNORM = ""          # "" for counts, or "probability" / "percent"
BARMODE = "overlay"    # "overlay" or "stack"
OPACITY = 0.55         # flat-ish fill


# -----------------------
# HELPERS
# -----------------------
def gait_metrics_csv_path(recording: Path, tracker: str) -> Path:
    if tracker == "qualisys":
        return recording / "validation" / tracker / "gait_parameters" / "qualisys_gait_metrics.csv"
    else:
        return recording / "validation" / tracker / "gait_parameters"/ "gait_metrics.csv"


def load_left_stance_pct(recording: Path, tracker: str) -> np.ndarray:
    fpath = gait_metrics_csv_path(recording, tracker)
    if not fpath.exists():
        raise FileNotFoundError(f"Missing: {fpath}")

    df = pd.read_csv(fpath)

    required = {"side", "metric", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{fpath} missing columns: {missing}")

    sub = df[(df["metric"] == METRIC) & (df["side"].astype(str).str.lower() == SIDE)].copy()
    y = pd.to_numeric(sub["value"], errors="coerce").to_numpy(dtype=float)
    y = y[np.isfinite(y)]
    return y


def darker(hex_color: str, factor: float = 0.65) -> str:
    """
    Return a darker shade of the given hex color.
    factor < 1 darkens; factor ~0.6-0.8 looks nice for edges.
    """
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    r = int(r * factor)
    g = int(g * factor)
    b = int(b * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


# -----------------------
# FIGURE: 3 plots (one per tracker), 5 hists each (one per session)
# -----------------------
fig = make_subplots(
    rows=1,
    cols=len(trackers),
    subplot_titles=[t for t in trackers],
    horizontal_spacing=0.06,
)

for col, tracker in enumerate(trackers, start=1):
    for i, recording in enumerate(recordings_list):
        # choose color by session index
        key = COLOR_ORDER[i % len(COLOR_ORDER)]
        fill = CONDITION_STYLE[key]["line"]
        edge = darker(fill, factor=0.65)

        label = mapping.get(recording.stem, recording.stem)

        try:
            y = load_left_stance_pct(recording, tracker)
        except Exception as e:
            print(f"[SKIP] {tracker} | {recording.stem}: {e}")
            continue

        if y.size == 0:
            print(f"[EMPTY] {tracker} | {recording.stem}: no {METRIC} for {SIDE}")
            continue

        fig.add_trace(
            go.Histogram(
                x=y,
                nbinsx=NBINS,
                histnorm=HISTNORM,
                name=f"{label}",
                marker=dict(
                    color=fill,
                    opacity=OPACITY,
                    line=dict(color=edge, width=1.2),
                ),
                showlegend=(col == 1),  # legend only once
            ),
            row=1,
            col=col,
        )

# Layout polish
fig.update_layout(
    title=f"Histogram of {METRIC} — {SIDE} leg (5 sessions per tracker)",
    template="plotly_white",
    barmode=BARMODE,
    height=420,
    width=1400,
    legend_title="Session (leg-length condition)",
)

# Axis titles
for c in range(1, len(trackers) + 1):
    fig.update_xaxes(title_text="Stance (%)", row=1, col=c)
fig.update_yaxes(title_text="Count" if HISTNORM == "" else HISTNORM, row=1, col=1)

fig.show()
