# ── user settings ────────────────────────────────────────────────────────────
BASE_DIR     = r"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3"
TRACKER_NAME = "mediapipe"
JOINT_NAME   = "knee_angle_r"
HEADER_ROWS  = 10
# 'pelvis_tilt', 'pelvis_list', 'pelvis_rotation', 'pelvis_tx',
#        'pelvis_ty', 'pelvis_tz', 'hip_flexion_r', 'hip_adduction_r',
#        'hip_rotation_r', 'knee_angle_r', 'knee_angle_r_beta', 'ankle_angle_r',
#        'subtalar_angle_r', 'mtp_angle_r', 'hip_flexion_l', 'hip_adduction_l',
#        'hip_rotation_l', 'knee_angle_l', 'knee_angle_l_beta', 'ankle_angle_l',
#        'subtalar_angle_l', 'mtp_angle_l', 'L5_S1_Flex_Ext',
#        'L5_S1_Lat_Bending', 'L5_S1_axial_rotation', 'L4_L5_Flex_Ext',
#        'L4_L5_Lat_Bending', 'L4_L5_axial_rotation', 'L3_L4_Flex_Ext',
#        'L3_L4_Lat_Bending', 'L3_L4_axial_rotation', 'L2_L3_Flex_Ext',
#        'L2_L3_Lat_Bending', 'L2_L3_axial_rotation', 'L1_L2_Flex_Ext',
#        'L1_L2_Lat_Bending', 'L1_L2_axial_rotation', 'L1_T12_Flex_Ext',
#        'L1_T12_Lat_Bending', 'L1_T12_axial_rotation', 'Abs_r3', 'Abs_r2',
#        'Abs_r1', 'Abs_t1', 'Abs_t2', 'neck_flexion', 'neck_bending',
#        'neck_rotation', 'arm_flex_r', 'arm_add_r', 'arm_rot_r', 'elbow_flex_r',
#        'pro_sup_r', 'wrist_flex_r', 'wrist_dev_r', 'arm_flex_l', 'arm_add_l',
#        'arm_rot_l', 'elbow_flex_l', 'pro_sup_l', 'wrist_flex_l',
#        'wrist_dev_l'

# ─────────────────────────────────────────────────────────────────────────────

from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BLUE = "rgba(31, 119, 180, 0.7)"   # FreeMoCap
RED  = "rgba(214, 39, 40, 0.7)"    # Qualisys
GREEN = "rgba(0, 255, 171, 0.7)"  # FreeMoCap Blender
BLACK = "rgba(0, 0, 0, 0.7)"      # FreeMoCap Blender
session_dirs = [
    Path(r"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_13_48_44_MDN_treadmill_2")
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
    fmc_blender_path = sesh / "sweep_angles_all.csv"
    if not (qual_path.exists() and fmc_path.exists()):
        print(f"[warning] Missing IK files in {sesh.name} – skipped.")
        continue

    qual = load_mot(qual_path, HEADER_ROWS)
    fmc  = load_mot(fmc_path,  HEADER_ROWS)
    fmc_blender = pd.read_csv(fmc_blender_path) if fmc_blender_path.exists() else None
    
    if JOINT_NAME not in qual.columns or JOINT_NAME not in fmc.columns:
        print(f"[warning] Joint '{JOINT_NAME}' not found in {sesh.name} – skipped.")
        continue

    qual_flipped = qual[JOINT_NAME]*-1

    # update global x-axis bounds
    tmin = min(qual.index.min(), fmc.index.min())
    tmax = max(qual.index.max(), fmc.index.max())
    global_tmin = tmin if global_tmin is None else min(global_tmin, tmin)
    global_tmax = tmax if global_tmax is None else max(global_tmax, tmax)

    ymin = min(fmc[JOINT_NAME].min(), (qual_flipped).min())   # *-1 if you keep that sign flip
    ymax = max(fmc[JOINT_NAME].max(), (qual_flipped).max())
    global_ymin = ymin if global_ymin is None else min(global_ymin, ymin) - 10
    global_ymax = ymax if global_ymax is None else max(global_ymax, ymax) + 10
    
    fig.add_trace(
        go.Scatter(
            x=fmc.index, y=qual_flipped,
            mode="lines", name="Qualisys" if idx == 1 else None,
            line=dict(color=BLACK, dash="dash"),
            legendgroup="qual", showlegend=idx == 1,
        ),
        row=idx, col=1,
    )
    # add traces
    fig.add_trace(
        go.Scatter(
            x=fmc.index, y=fmc[JOINT_NAME],
            mode="lines", name=f"FreeMoCap – {TRACKER_NAME} - OpenSim" if idx == 1 else None,
            line=dict(color=BLUE),
            legendgroup="fmc", showlegend=idx == 1,
        ),
        row=idx, col=1,
    )


    fig.add_trace(
        go.Scatter(
            x=fmc.index, y=fmc_blender['angle_angle#right_knee_extension_flexion'],
            mode="lines", name="FreeMoCap – Blender" if idx == 1 else None,
            line=dict(color=RED),
            legendgroup="fmc_blender", showlegend=idx == 1,
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
