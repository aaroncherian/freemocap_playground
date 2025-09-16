from skellymodels.models.tracking_model_info import ModelInfo
from skellymodels.managers.human import Human
from skellymodels.managers.animal import Animal
from skellymodels.models.trajectory import Trajectory
from skellymodels.models.aspect import TrajectoryNames
import logging
from pathlib import Path
import numpy as np

# ---------- core Kabsch ----------
def kabsch(P0, P, w=None):
    if w is None:
        w = np.ones(len(P0), dtype=float)
    w = w.reshape(-1, 1)
    W = w.sum()

    c0 = (w * P0).sum(axis=0) / W
    c  = (w * P ).sum(axis=0) / W

    X0 = (P0 - c0) * w
    X  = (P  - c)

    H = X.T @ X0
    U, S, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    t = c - R @ c0
    return R, t

# ---------- robust weights (softly down-weights outliers like a jumping toe) ----------
def huber_weights(res_mm, k_mm=25.0):
    r = np.asarray(res_mm, dtype=float)
    w = np.ones_like(r)
    m = r > k_mm
    # w = k/|r| for outliers
    w[m] = k_mm / np.maximum(r[m], 1e-9)
    return w

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- paths/inputs ---
path_to_recording = Path(r'D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_11_55_05_TF01_flexion_neg_5_6_trial_1')
neutral_frames = slice(100, 250)
output_data_folder = path_to_recording/'output_data'/'dlc_rigidified'
output_data_folder.mkdir(parents=True, exist_ok=True)

human: Human = Human.from_data(path_to_recording / 'output_data' / 'mediapipe_dlc')

# Marker name -> index
leg_names = ['right_knee', 'right_ankle', 'right_heel', 'right_foot_index']
name2idx = {n: human.body.anatomical_structure.landmark_names.index(n) for n in leg_names}
idx_knee = name2idx['right_knee']
idx_ankle = name2idx['right_ankle']
idx_heel = name2idx['right_heel']
idx_toe  = name2idx['right_foot_index']

# Slice out the 4 markers (T,4,3)
P_frames = human.body.xyz.as_array[:, [idx_knee, idx_ankle, idx_heel, idx_toe], :].astype(float)
T = P_frames.shape[0]

# Define segment sets (indices into the 4-marker slice)
# shank uses knee, ankle, heel; foot uses ankle, heel, toe
SHANK_IDS = np.array([0, 1, 2])  # knee, ankle, heel
FOOT_IDS  = np.array([1, 2, 3])  # ankle, heel, toe

# Build templates from neutral window (nan-robust)
P0_shank = np.nanmedian(P_frames[neutral_frames][:, SHANK_IDS, :], axis=0)  # (3,3)
P0_foot  = np.nanmedian(P_frames[neutral_frames][:, FOOT_IDS,  :], axis=0)  # (3,3)

# Sanity logs (optional)
def _dist(P0, a, b): return np.linalg.norm(P0[a] - P0[b])
logger.info(f"Shank template distances (m): knee-ankle={_dist(P0_shank,0,1):.3f}, ankle-heel={_dist(P0_shank,1,2):.3f}")
logger.info(f"Foot  template distances (m): ankle-heel={_dist(P0_foot,0,1):.3f}, heel-toe={_dist(P0_foot,1,2):.3f}")

# Outputs
P_fixed = np.copy(P_frames)                 # (T,4,3)
R_shank_list = np.tile(np.eye(3), (T,1,1))
t_shank_list = np.zeros((T,3))
R_foot_list  = np.tile(np.eye(3), (T,1,1))
t_foot_list  = np.zeros((T,3))
residuals_mm = np.full((T, 4), np.nan)

# Last-good fallbacks per segment
R_shank_last = np.eye(3); t_shank_last = np.zeros(3)
R_foot_last  = np.eye(3); t_foot_last  = np.zeros(3)

# Units: Freemocap world is usually meters
SCALE_MM = 1000.0

# --- per-frame fits ---
for k in range(T):
    frame = P_frames[k]  # (4,3)

    # ---- SHANK fit (knee, ankle, heel) ----
    obs_shank = frame[SHANK_IDS]                         # (3,3)
    valid_shank = np.isfinite(obs_shank).all(axis=1)     # (3,)
    if valid_shank.sum() >= 3:
        R_s, t_s = kabsch(P0_shank[valid_shank], obs_shank[valid_shank])
        R_shank_last, t_shank_last = R_s, t_s
    else:
        R_s, t_s = R_shank_last, t_shank_last
    pred_shank = P0_shank @ R_s.T + t_s                  # (3,3)

    # ---- FOOT fit (ankle, heel, toe) with Huber reweight on residuals ----
    obs_foot  = frame[FOOT_IDS]
    valid_foot = np.isfinite(obs_foot).all(axis=1)
    if valid_foot.sum() >= 3:
        # initial equal-weight fit
        R_f, t_f = kabsch(P0_foot[valid_foot], obs_foot[valid_foot])

        # residuals (for robust weights)
        pred0 = P0_foot @ R_f.T + t_f
        r = np.full(3, np.nan)
        r[valid_foot] = np.linalg.norm(obs_foot[valid_foot] - pred0[valid_foot], axis=1)
        w = np.ones(3)
        w[valid_foot] = huber_weights(r[valid_foot] * SCALE_MM, k_mm=25.0)

        # refit with Huber weights
        R_f, t_f = kabsch(P0_foot[valid_foot], obs_foot[valid_foot], w=w[valid_foot])
        R_foot_last, t_foot_last = R_f, t_f
    else:
        R_f, t_f = R_foot_last, t_foot_last
    pred_foot = P0_foot @ R_f.T + t_f

    # ---- write back rigidified markers, resolving overlap ----
    # knee from shank, ankle/heel/toe from foot
    P_fixed[k, 0, :] = pred_shank[0]   # knee
    P_fixed[k, 1, :] = pred_foot[0]    # ankle
    P_fixed[k, 2, :] = pred_foot[1]    # heel
    P_fixed[k, 3, :] = pred_foot[2]    # toe

    # ---- store transforms and residuals (diagnostics) ----
    R_shank_list[k] = R_s; t_shank_list[k] = t_s
    R_foot_list[k]  = R_f; t_foot_list[k]  = t_f

    # residuals vs observed where present (mm)
    ok = np.isfinite(frame).all(axis=1)
    pred_full = np.vstack([pred_shank[0], pred_foot])    # rows: knee, ankle, heel, toe
    residuals_mm[k, ok] = np.linalg.norm(frame[ok] - pred_full[ok], axis=1) * SCALE_MM

# ---- splice back into the full skeleton (overwrite the four markers) ----
human.body.xyz.as_array[:, [idx_knee, idx_ankle, idx_heel, idx_toe], :] = P_fixed

# ---- save outputs just like before ----
human.save_out_all_data_csv(output_data_folder)
human.save_out_all_xyz_numpy_data(output_data_folder)
human.save_out_all_data_parquet(output_data_folder)
human.save_out_csv_data(output_data_folder)
human.save_out_numpy_data(output_data_folder)
