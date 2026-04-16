from skellymodels.managers.human import Human
from skellymodels.models.tracking_model_info import MediapipeModelInfo

from scipy.signal import find_peaks

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from dataclasses import dataclass
import pandas as pd

def calculate_spherical_angles(human: Human):
    body_3d_xyz = human.body.trajectories['3d_xyz']
    spine_vector = body_3d_xyz.segment_data(human.body.anatomical_structure.segment_connections)['spine']['proximal'] - body_3d_xyz.segment_data(human.body.anatomical_structure.segment_connections)['spine']['distal']
    spine_vector_magnitude = np.linalg.norm(spine_vector, axis=1)
    spine_vector_azimuthal = np.arctan2(spine_vector[:, 1], spine_vector[:, 0])
    spine_vector_polar = np.arccos(spine_vector[:,2]/(spine_vector_magnitude + 1e-9))

    return spine_vector_azimuthal, spine_vector_polar, spine_vector_magnitude


def calculate_trunk_inclination(path_to_recording:Path):
    path_to_output_data = path_to_recording / 'output_data'/     'mediapipe_skeleton_3d.npy'

    human = Human.from_tracked_points_numpy_array(
        name = 'human', 
        model_info=MediapipeModelInfo(),
        tracked_points_numpy_array=np.load(path_to_output_data)
    )
    spine_vector_azimuthal, spine_vector_polar, spine_vector_magnitude = calculate_spherical_angles(
        human=human
    )
    spine_vector_polar = np.degrees(spine_vector_polar)

    return spine_vector_polar

@dataclass
class AnglesData:
    angles: np.ndarray
    peak_indices: np.ndarray
    peak_angles: np.ndarray


semi_squat_paths = {...}


#you can download data from the NAS and extract it to D:\Matthis\lifting_data - delete it from the C:\Downloads folder afterwards (and empty the recyling bin)

path_dicts = {
    "GB": {
        "squat": r"D:\Matthis\lifting_data\2026-02-08_14-50-33_GMT-5_GB_squat",
        "stoop": r"D:\Matthis\lifting_data\2026-02-08_14-53-01_GMT-5_GB_stoop",
        "free": r"D:\Matthis\lifting_data\2026-02-08_14-57-56_GMT-5_GB_free"
    },
    "OK": {
        
    }


}


angles_dict = {}

rows_angle_data = []

for participant, data_paths in path_dicts.items():
    for condition, data_path in data_paths.items():
        if not data_path: 
            continue
        angles = calculate_trunk_inclination(path_to_recording=Path(data_path))
        peaks, _ = find_peaks(angles, height=20, distance=25)
        peak_angles = angles[peaks]
        
        angle_row = {
            "participant": participant,
            "condition": condition,
            "angles": angles,
            "peak_indices": peaks,
            "peak_angles": peak_angles
            }
        rows_angle_data.append(angle_row)

        f =2
        
angle_df = pd.DataFrame(rows_angle_data)

path_to_save_folder = Path(r"D:\Matthis\lifting_data")

angle_df.to_csv(path_to_save_folder/"angle_data.csv")

SHOW_PLOT = True

# ---- Plotting: one figure per participant, shared y-axis limits ----
for participant, participant_df in angle_df.groupby("participant"):
    n_conditions = len(participant_df)
    
    fig, axes = plt.subplots(n_conditions, 1, figsize=(10, 4 * n_conditions), sharex=True)
    
    if n_conditions == 1:
        axes = [axes]
    
    # Compute global y-limits for this participant
    all_angles = np.concatenate(participant_df["angles"].values)
    ymin, ymax = all_angles.min(), all_angles.max()
    
    for ax, (_, row) in zip(axes, participant_df.iterrows()):
        condition = row["condition"]
        angles = row["angles"]
        peaks = row["peak_indices"]
        
        ax.plot(angles, label=f"{condition}", color='blue')
        ax.scatter(peaks, angles[peaks], color='red', marker='o', label="peaks")
        
        ax.set_title(f"{participant} - {condition}")
        ax.set_ylabel("Angle (deg)")
        ax.set_ylim(ymin, ymax)  # <-- enforce same scale
        ax.grid(True)
        ax.legend()
    
    axes[-1].set_xlabel("Frame")
    
    plt.savefig(path_to_save_folder/f"{participant}_trunk_inclination.png")
    plt.tight_layout()

    if SHOW_PLOT:
        plt.show()



#the dataframe is set up as the participant ID, the condition name, an array of the trunk inclination angles, and an array of where the peak angles occurred, an array of the peak angle values 

#angle_df.query("participant == 'OK' and condition == 'stoop'")["angles"]

f = 2





# for name, path_to_recording in data_folders.items():
#     angles = calculate_trunk_inclination(path_to_recording=path_to_recording)
#     peaks, _ = find_peaks(angles, height=20, distance=25)
#     peak_angles = angles[peaks]
    
#     angles_dict[name] = AnglesData(
#         angles = angles,
#         peak_indices = peaks,
#         peak_angles = peak_angles
#     )

# f = 2

# fig, (ax1, ax2) = plt.subplots(1,2)

# stoop_data: AnglesData = angles_dict["KK_stoop"]
# squat_data: AnglesData = angles_dict["KK_squat"]


# ax1.plot(stoop_data.angles)
# ax2.plot(squat_data.angles)

# ylim = 100
# ax1.set_ylim([0,ylim])
# ax2.set_ylim([0,ylim])

# ax1.scatter(stoop_data.peak_indices, stoop_data.peak_angles, color = 'red')
# ax2.scatter(squat_data.peak_indices, squat_data.peak_angles, color = 'red')

# ax1.set_title("Stoop Form")
# ax2.set_title("Squat Form")

# ax1.set_xlabel("Time (frames)")
# ax2.set_xlabel("Time (frames)")

# ax1.set_ylabel("Trunk inclination (degrees)")

# plt.show()

# max_peaks = max(len(d.peak_angles) for d in angles_dict.values())

# data = {}
# for trial_name, d in angles_dict.items():
#     col = np.full(max_peaks, "", dtype=object)
#     col[:len(d.peak_angles)] = d.peak_angles.astype(float)
#     data[trial_name] = col

# df = pd.DataFrame(data)
# df.insert(0, "peak_number", np.arange(1, max_peaks + 1))

# df.to_csv(r"D:\lifting_pilot\trunk_inclination_peaks_comparison.csv", index=False)
f = 2

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # -----------------------------
# # Wide -> long (tidy) table
# # -----------------------------
# peak_cols = [c for c in df.columns if c != "peak_number"]

# df_long = df.melt(
#     id_vars="peak_number",
#     value_vars=peak_cols,
#     var_name="trial",
#     value_name="peak_angle",
# ).copy()

# # your padding is "" -> coerce to NaN then drop
# df_long["peak_angle"] = pd.to_numeric(df_long["peak_angle"], errors="coerce")
# df_long = df_long.dropna(subset=["peak_angle"])

# # parse trial names like "OK_squat"
# df_long["participant"] = df_long["trial"].str.split("_").str[0]
# df_long["form"] = df_long["trial"].str.split("_").str[1]

# # enforce order
# form_order = ["squat", "stoop"]
# df_long = df_long[df_long["form"].isin(form_order)].copy()

# # -----------------------------
# # Form-level strip plot
# # -----------------------------
# rng = np.random.default_rng(0)  # reproducible jitter

# fig, ax = plt.subplots(figsize=(7, 5))

# xpos = {form: i for i, form in enumerate(form_order)}

# # color by participant (stable order)
# participants = sorted(df_long["participant"].unique())
# colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
# color_map = {p: colors[i % len(colors)] for i, p in enumerate(participants)} if colors else {}

# for p in participants:
#     sub = df_long[df_long["participant"] == p]
#     x = sub["form"].map(xpos).to_numpy(dtype=float)
#     y = sub["peak_angle"].to_numpy(dtype=float)
#     jitter = rng.uniform(-0.18, 0.18, size=len(sub))
#     ax.scatter(x + jitter, y, alpha=0.85, label=p, c=color_map.get(p, None))

# # overlay form means as thick underscores
# form_means = df_long.groupby("form")["peak_angle"].mean().reindex(form_order)
# ax.scatter(
#     [xpos[f] for f in form_means.index],
#     form_means.values,
#     s=250,
#     marker="_",
#     linewidths=4,
#     c="black",
#     zorder=5,
# )

# ax.set_xticks(range(len(form_order)))
# ax.set_xticklabels([f.title() for f in form_order])
# ax.set_ylabel("Trunk inclination peak (deg)")
# ax.set_title("Peak trunk inclination by form (all peaks)")
# ax.set_ylim(0, 110)  # tweak as needed
# ax.grid(True, axis="y", alpha=0.3)
# ax.legend(title="Participant", frameon=False)

# plt.tight_layout()
# plt.show()

# # -----------------------------
# # Optional: Form-level box plot
# # -----------------------------
# fig, ax = plt.subplots(figsize=(7, 5))
# data_by_form = [df_long.loc[df_long["form"] == f, "peak_angle"].to_numpy() for f in form_order]
# ax.boxplot(data_by_form, labels=[f.title() for f in form_order], showfliers=True)
# ax.set_ylabel("Trunk inclination peak (deg)")
# ax.set_title("Peak trunk inclination by form (box plot)")
# ax.set_ylim(0, 110)
# ax.grid(True, axis="y", alpha=0.3)
# plt.tight_layout()
# plt.show()
