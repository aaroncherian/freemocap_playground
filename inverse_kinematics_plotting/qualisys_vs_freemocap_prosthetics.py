# ── user settings ────────────────────────────────────────────────────────────
BASE_DIR     = r"D:\2023-06-07_TF01\1.0_recordings\treadmill_calib"
TRACKER_NAME = "mediapipe_dlc"
JOINT_NAME   = "ankle_angle_r"
HEADER_ROWS  = 10
# ─────────────────────────────────────────────────────────────────────────────

from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BLUE = "rgba(31, 119, 180, 0.7)"   # FreeMoCap
RED  = "rgba(214, 39, 40, 0.7)"    # Qualisys

session_dirs = [
    # Path(r"D:\2023-06-07_TF01\1.0_recordings\treadmill_calib\sesh_2023-06-07_11_55_05_TF01_flexion_neg_5_6_trial_1"),
    Path(r"D:\2023-06-07_TF01\1.0_recordings\treadmill_calib\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1"),
    # Path(r"D:\2023-06-07_TF01\1.0_recordings\treadmill_calib\sesh_2023-06-07_12_12_36_TF01_flexion_pos_5_6_trial_1"),
]

def load_mot(path: Path, n_header: int) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        delim_whitespace=True,
        skiprows=n_header,
        comment="#",
        header=0,
    )
    df.rename(columns={df.columns[0]: "time"}, inplace=True)
    return df.set_index("time")

# --------------------------------------------------------------------------- #
print(f"Found {len(session_dirs)} session(s). Building figure …")

fig = make_subplots(
    rows=len(session_dirs),
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04,
    subplot_titles=[s.name for s in session_dirs],
)
global_tmin, global_tmax = None, None
global_ymin, global_ymax = None, None


for idx, sesh in enumerate(session_dirs, 1):
    qual_path = sesh / "validation" / "qualisys" / "qualisys_ik_results.mot"
    fmc_path  = sesh / "validation" / TRACKER_NAME / f"{TRACKER_NAME}_ik_results.mot"
    if not (qual_path.exists() and fmc_path.exists()):
        print(f"[warning] Missing IK files in {sesh.name} – skipped.")
        continue

    qual = load_mot(qual_path, HEADER_ROWS)
    fmc  = load_mot(fmc_path,  HEADER_ROWS)
    if JOINT_NAME not in qual.columns or JOINT_NAME not in fmc.columns:
        print(f"[warning] Joint '{JOINT_NAME}' not found in {sesh.name} – skipped.")
        continue

    # update global x-axis bounds
    tmin = min(qual.index.min(), fmc.index.min())
    tmax = max(qual.index.max(), fmc.index.max())
    global_tmin = tmin if global_tmin is None else min(global_tmin, tmin)
    global_tmax = tmax if global_tmax is None else max(global_tmax, tmax)

    ymin = min(fmc[JOINT_NAME].min(), (qual[JOINT_NAME]*-1).min())   # *-1 if you keep that sign flip
    ymax = max(fmc[JOINT_NAME].max(), (qual[JOINT_NAME]*-1).max())
    global_ymin = ymin if global_ymin is None else min(global_ymin, ymin)
    global_ymax = ymax if global_ymax is None else max(global_ymax, ymax) + 10

    # add traces
    fig.add_trace(
        go.Scatter(
            x=fmc.index, y=fmc[JOINT_NAME],
            mode="lines", name=f"FreeMoCap – {TRACKER_NAME}" if idx == 1 else None,
            line=dict(color=BLUE),
            legendgroup="fmc", showlegend=idx == 1,
        ),
        row=idx, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=qual.index, y=qual[JOINT_NAME],
            mode="lines", name="Qualisys" if idx == 1 else None,
            line=dict(color=RED, dash="dash"),
            legendgroup="qual", showlegend=idx == 1,
        ),
        row=idx, col=1,
    )

# --------------------------------------------------------------------------- #
fig.update_layout(
    height=300 * len(session_dirs) + 120,
    width=3000,               # wider canvas
    title=dict(
        text=f"{JOINT_NAME} • {TRACKER_NAME} vs Qualisys (all sessions)",
        x=0.5, xanchor="center",
    ),
    template="plotly_white",
    xaxis_title="Time (s)",
    yaxis_title="Angle (deg)",
)

# lock the same x-axis range on every subplot
fig.update_xaxes(
    range=[global_tmin, global_tmax],
    matches="x"              # keeps all rows in lock-step
)

fig.update_yaxes(
    range=[global_ymin, global_ymax],
    matches="y"          # any zoom on one row now moves all rows
)

# give every subplot a y-axis label
for r in range(1, len(session_dirs) + 1):
    fig["layout"][f"yaxis{r}"]["title"] = "Angle (deg)"

fig.show()

# optional standalone HTML
out_html = Path(BASE_DIR) / f"{JOINT_NAME}_{TRACKER_NAME}_vs_qualisys_all_sessions.html"
fig.write_html(out_html)
print(f"✔ saved HTML to:\n{out_html}")
