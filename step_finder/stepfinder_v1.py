from skellymodels.managers.human import Human
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from skellyforge.freemocap_utils.postprocessing_widgets.postprocessing_functions.filter_data import butter_lowpass_filter
import logging
from dataclasses import dataclass
import pandas as pd


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GaitEvents:
    heel_strikes: np.ndarray
    toe_offs: np.ndarray

@dataclass
class GaitResults:
    right_foot: GaitEvents
    left_foot: GaitEvents

def get_velocity(positions:np.ndarray, sampling_rate:float):
    dt = 1.0 / sampling_rate
    velocities = np.gradient(positions, dt, axis=0)
    return velocities

def remove_events_within_minimum_interval(event_candidates:np.ndarray, min_interval:float, sampling_rate:float):
    dt = 1/sampling_rate
    events = []

    i = 0
    logger.info(f"Removing events that are within {min_interval:.2f}s of each other from {event_candidates.shape[0]} candidates")

    events = [event_candidates[0]]
    for i in event_candidates[1:]:
        if (i - events[-1])*dt > min_interval:
            events.append(i)
        else:
            logger.info(f"Removing event at frame {i} which is {(i - events[-1])*dt:.3f}s after previous event at frame {events[-1]}")

    return np.array(events, dtype = int)

def get_heel_strike_and_toe_off_events(heel_velocity:np.ndarray, toe_velocity:np.ndarray, min_interval_s:float=0.25, sampling_rate:float=30.0):
    heel_strike_candidates = np.where((heel_velocity[:-1,1]>0) & (heel_velocity[1:, 1] <= 0)) [0] + 1
    toe_off_candidates = np.where((toe_velocity[:-1,1]<=0) & (toe_velocity[1:,1]>0))[0] + 1
    
    heel_strikes = remove_events_within_minimum_interval(
        event_candidates=heel_strike_candidates,
        min_interval=min_interval_s,
        sampling_rate=sampling_rate
    )
    toe_offs = remove_events_within_minimum_interval(
        event_candidates=toe_off_candidates,
        min_interval=min_interval_s,
        sampling_rate=sampling_rate
    )
    return GaitEvents(heel_strikes=heel_strikes, toe_offs=toe_offs)

def detect_gait_events(human:Human, sampling_rate:float=30.0):

    left_heel = human.body.xyz.as_dict['left_heel']
    right_heel = human.body.xyz.as_dict['right_heel']
    
    left_toe = human.body.xyz.as_dict['left_foot_index']
    right_toe = human.body.xyz.as_dict['right_foot_index']

    left_heel_velocity = get_velocity(left_heel, sampling_rate)
    right_heel_velocity = get_velocity(right_heel, sampling_rate)
    left_toe_velocity = get_velocity(left_toe, sampling_rate)
    right_toe_velocity = get_velocity(right_toe, sampling_rate)

    right_foot_gait_events:GaitEvents = get_heel_strike_and_toe_off_events(
        heel_velocity=right_heel_velocity,
        toe_velocity=right_toe_velocity,
        sampling_rate = sampling_rate)
    
    left_foot_gait_events:GaitEvents = get_heel_strike_and_toe_off_events(
        heel_velocity=left_heel_velocity,
        toe_velocity=left_toe_velocity,
        sampling_rate = sampling_rate)

    return GaitResults(right_foot=right_foot_gait_events, left_foot=left_foot_gait_events)



import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ---------- small helpers ----------
def _lp(x, fs, fc, order=4, axis=0):
    b, a = butter(order, fc/(0.5*fs), btype="low")
    return filtfilt(b, a, x, axis=axis)

def _summarize_cycles(hs_idx, to_idx, fs):
    hs = np.asarray(hs_idx); to = np.asarray(to_idx)
    strides, stances, swings, stance_pc = [], [], [], []
    for k in range(len(hs)-1):
        hs0, hs1 = hs[k], hs[k+1]
        # find first TO between these HS
        mids = to[(to > hs0) & (to < hs1)]
        if len(mids) == 0: 
            continue
        to0 = mids[0]
        stride = (hs1 - hs0)/fs
        stance = (to0 - hs0)/fs
        swing  = stride - stance
        strides.append(stride); stances.append(stance); swings.append(swing)
        stance_pc.append(100.0*stance/stride)
    return (np.array(strides), np.array(stances), np.array(swings), np.array(stance_pc))

def _print_summary(foot_label, hs_idx, to_idx, fs):
    strides, stances, swings, stance_pc = _summarize_cycles(hs_idx, to_idx, fs)
    print(f"\n[{foot_label}] n cycles: {len(strides)}")
    if len(strides):
        print(f"Stride time (s):  {strides.mean():.3f} ± {strides.std():.3f}")
        print(f"Stance time (s):  {stances.mean():.3f} ± {stances.std():.3f}")
        print(f"Stance %   (%):   {stance_pc.mean():.1f} ± {stance_pc.std():.1f}")
        print(f"Cadence (spm):   {2*60.0/strides.mean():.1f}")  # steps/min

# ---------- make display signals (filtered pos -> vel) ----------
def _prepare_signals_for_qc(heel_xyz, toe_xyz, fs, ap_axis=1, z_axis=2, pos_fc=3.0, vel_fc=6.0):
    # Filter positions lightly (display only)
    heel_pos = _lp(heel_xyz, fs, pos_fc, order=4, axis=0)
    toe_pos  = _lp(toe_xyz,  fs, pos_fc, order=4, axis=0)
    # Differentiate -> velocity and lightly LP
    dt = 1.0/fs
    heel_v = _lp(np.gradient(heel_pos, dt, axis=0), fs, vel_fc, order=2, axis=0)
    toe_v  = _lp(np.gradient(toe_pos,  dt, axis=0), fs, vel_fc, order=2, axis=0)
    # Components we want
    heel_v_ap = heel_v[:, ap_axis]
    toe_v_ap  = toe_v[:,  ap_axis]
    heel_z    = heel_pos[:, z_axis]
    toe_z     = toe_pos[:,  z_axis]
    return heel_v_ap, toe_v_ap, heel_z, toe_z

# ---------- main QC plotter for one foot ----------
def plot_gait_qc_for_foot(heel_xyz, toe_xyz, hs_idx, to_idx, fs,
                          ap_axis=1, z_axis=2, pos_fc=3.0, vel_fc=6.0,
                          foot_label="right"):
    heel_v_ap, toe_v_ap, heel_z, toe_z = _prepare_signals_for_qc(
        heel_xyz, toe_xyz, fs, ap_axis, z_axis, pos_fc, vel_fc
    )
    t = np.arange(len(heel_v_ap))/fs

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(f"Zeni-style treadmill detector: {foot_label} foot — AP velocities + events")

    # (1) Heel AP velocity + HS
    axes[0].plot(t, heel_v_ap, label="Heel AP velocity (filtered)")
    axes[0].axhline(0, color='k', lw=0.6)
    if len(hs_idx):
        axes[0].plot(hs_idx/fs, heel_v_ap[hs_idx], 'go', label="HS")
    axes[0].set_ylabel("AP vel (units/s)")
    axes[0].legend(loc="upper right")

    # (2) Toe AP velocity + TO
    axes[1].plot(t, toe_v_ap, label="Toe AP velocity (filtered)")
    axes[1].axhline(0, color='k', lw=0.6)
    if len(to_idx):
        axes[1].plot(to_idx/fs, toe_v_ap[to_idx], 'ro', label="TO")
    axes[1].set_ylabel("AP vel (units/s)")
    axes[1].legend(loc="upper right")

    # (3) Vertical heel/toe + both events
    axes[2].plot(t, heel_z, label="Heel z (filtered)")
    axes[2].plot(t, toe_z,  label="Toe z (filtered)", alpha=0.85)
    if len(hs_idx):
        axes[2].plot(hs_idx/fs, heel_z[hs_idx], 'go', ms=6, label="HS")
    if len(to_idx):
        axes[2].plot(to_idx/fs,  toe_z[to_idx],  'ro', ms=6, label="TO")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Vertical (units)")
    axes[2].legend(loc="upper right")

    plt.tight_layout()
    plt.show()

    _print_summary(foot_label, hs_idx, to_idx, fs)

# ---------- drive it with your existing detector ----------
def qc_both_feet(human, sampling_rate=30.0):
    # your detector
    results = detect_gait_events(human, sampling_rate)

    # right foot
    r_heel = human.body.xyz.as_dict['right_heel']
    r_toe  = human.body.xyz.as_dict['right_foot_index']
    plot_gait_qc_for_foot(r_heel, r_toe, 
                          results.right_foot.heel_strikes, results.right_foot.toe_offs,
                          fs=sampling_rate, foot_label="right")

    # left foot
    l_heel = human.body.xyz.as_dict['left_heel']
    l_toe  = human.body.xyz.as_dict['left_foot_index']
    plot_gait_qc_for_foot(l_heel, l_toe, 
                          results.left_foot.heel_strikes, results.left_foot.toe_offs,
                          fs=sampling_rate, foot_label="left")


def save_gait_events_to_csv(results:GaitResults, fs:float, out_path:Path):
    """Save gait events (frames + times) for both feet into a CSV."""
    rows = []

    for foot_label, events in [
        ("right", results.right_foot),
        ("left", results.left_foot),
    ]:
        for idx in events.heel_strikes:
            rows.append({"foot": foot_label, "event": "heel_strike",
                         "frame": int(idx), "time_s": idx/fs})
        for idx in events.toe_offs:
            rows.append({"foot": foot_label, "event": "toe_off",
                         "frame": int(idx), "time_s": idx/fs})

    df = pd.DataFrame(rows)
    df = df.sort_values(["time_s"]).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    print(f"Saved gait events to {out_path}")


path_to_recording = Path(r'D:\2025_07_31_JSM_pilot\freemocap\2025-07-31_16-35-10_GMT-4_jsm_treadmill_trial_1')

path_to_data = path_to_recording/'validation'/'qualisys'

human:Human = Human.from_data(path_to_data)
gait_events = detect_gait_events(human, sampling_rate=30.0)
qc_both_feet(human, sampling_rate=30.0)
save_gait_events_to_csv(gait_events, fs=30.0, out_path=path_to_data/'gait_events.csv')
# gait_events = detect_gait_events(human)

# # --- QC PLOTS ---
# t = np.arange(len(heel_v_ap_f)) / fs
# import matplotlib.pyplot as plt

# fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

# # (1) Heel AP velocity + HS
# axes[0].plot(t, heel_v_ap_f, label='Heel AP velocity (filtered)')
# axes[0].axhline(0, color='k', lw=0.5)
# axes[0].plot(hs_idx/fs, heel_v_ap_f[hs_idx], 'go', label='HS')
# axes[0].set_ylabel('AP vel (units/s)')
# axes[0].legend(loc='upper right')
# axes[0].set_title('Zeni-style treadmill detector: AP velocities + events')

# # (2) Toe AP velocity + TO
# axes[1].plot(t, toe_v_ap_f, label='Toe AP velocity (filtered)')
# axes[1].axhline(0, color='k', lw=0.5)
# axes[1].plot(to_idx/fs, toe_v_ap_f[to_idx], 'ro', label='TO')
# axes[1].set_ylabel('AP vel (units/s)')
# axes[1].legend(loc='upper right')

# # (3) Vertical positions (heel/toe) with both events
# axes[3-1].plot(t, heel_z_f, label='Heel z (filtered)')
# axes[3-1].plot(t, toe_z_f,  label='Toe z (filtered)', alpha=0.8)
# axes[3-1].plot(hs_idx/fs, heel_z_f[hs_idx], 'go', ms=6, label='HS')
# axes[3-1].plot(to_idx/fs,  toe_z_f[to_idx],  'ro', ms=6, label='TO')
# axes[3-1].set_xlabel('Time (s)')
# axes[3-1].set_ylabel('Vertical (units)')
# axes[3-1].legend(loc='upper right')

# plt.tight_layout()
# plt.show()

# # --- QUICK METRICS ---
# # stride = HS -> next HS, stance = HS -> TO, swing = TO -> next HS
# def _pairs(hs, to):
#     hs = np.asarray(hs); to = np.asarray(to)
#     strides   = []
#     stances   = []
#     swings    = []
#     stance_pc = []
#     for k in range(len(hs)-1):
#         hs0, hs1 = hs[k], hs[k+1]
#         # find first TO between them
#         within = to[(to > hs0) & (to < hs1)]
#         if len(within) == 0:
#             continue
#         to0 = within[0]
#         stride = (hs1 - hs0) / fs
#         stance = (to0 - hs0) / fs
#         swing  = stride - stance
#         strides.append(stride)
#         stances.append(stance)
#         swings.append(swing)
#         stance_pc.append(100.0 * stance / stride)
#     return np.array(strides), np.array(stances), np.array(swings), np.array(stance_pc)

# strides, stances, swings, stance_pc = _pairs(hs_idx, to_idx)

# print(f"n cycles: {len(strides)}")
# if len(strides):
#     print(f"Stride time (s):  mean {strides.mean():.3f} ± {strides.std():.3f}")
#     print(f"Stance time (s):  mean {stances.mean():.3f} ± {stances.std():.3f}")
#     print(f"Stance %  (%):    mean {stance_pc.mean():.1f} ± {stance_pc.std():.1f}")
#     print(f"Cadence (spm):    mean {60.0/strides.mean():.1f}")