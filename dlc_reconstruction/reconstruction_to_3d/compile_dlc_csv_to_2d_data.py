import pandas as pd
import numpy as np
from pathlib import Path

def compile_dlc_csvs(path_to_folder_with_dlc_csvs:Path,
                     confidence_threshold:float = 0.5,
                     interpolate = False
                     ):


    # Filtered csv list
    csv_list = sorted(list(path_to_folder_with_dlc_csvs.glob('*.csv')))

    # Initialize an empty list to hold dataframes
    dfs = []

    for csv in csv_list:
        # Read each csv into a dataframe with a multi-index header
        df = pd.read_csv(csv, header=[1, 2])
        
        # Drop the first column (which just has the headers )
        df = df.iloc[:, 1:]
        
        # Check if data shape is as expected
        if df.shape[1] % 3 != 0:
            print(f"Unexpected number of columns in {csv}: {df.shape[1]}")
            continue
        
        try:
            # Convert the df into a 4D numpy array of shape (1, num_frames, num_markers, 3) and append to dfs
            dfs.append(df.values.reshape(1, df.shape[0], df.shape[1]//3, 3))
        except ValueError as e:
            print(f"Reshape failed for {csv} with shape {df.shape}: {e}")


    # Concatenate all the arrays along the first axis (camera axis)
    dlc_2d_array_with_confidence = np.concatenate(dfs, axis=0)

    confidence_thresholded_dlc_2d_array_XYC = apply_confidence_threshold(array=dlc_2d_array_with_confidence, threshold=confidence_threshold)

    # final_thresholded_array = apply_confidence_threshold(final_array, 0.6)

    confidence_thresholded_dlc_2d_array_XY = confidence_thresholded_dlc_2d_array_XYC[:,:,:,:2]

    pre_interp_xy = confidence_thresholded_dlc_2d_array_XY.copy()

    if interpolate:
        confidence_thresholded_dlc_2d_array_XY = linear_interpolate_2d(
            confidence_thresholded_dlc_2d_array_XY
        )

     
    post_interp_xy = confidence_thresholded_dlc_2d_array_XY   
    import matplotlib.pyplot as plt
            # --- quick debug plot ---
    cam_idx = 2
    n_frames = post_interp_xy.shape[1]
    n_markers = post_interp_xy.shape[2]
    x_axis = np.arange(n_frames)

   
    n_frames = post_interp_xy.shape[1]
    n_markers = post_interp_xy.shape[2]
    x_axis = np.arange(n_frames)

    fig, axes = plt.subplots(n_markers, 2, figsize=(12, 3 * n_markers), sharex=True)
    if n_markers == 1:
        axes = np.array([axes])

    for m in range(n_markers):
        for dim, colname in enumerate(["X", "Y"]):
            ax = axes[m, dim]

            pre = pre_interp_xy[cam_idx, :, m, dim].astype(float)
            post = post_interp_xy[cam_idx, :, m, dim].astype(float)

            # masks
            pre_finite = np.isfinite(pre)
            filled = ~pre_finite & np.isfinite(post)     # points created by interpolation

            # 1) interpolated line
            ax.plot(x_axis, post, '-', color='blue', alpha=0.7, lw=1.25, label='Interpolated')

            # 2) original finite points only
            ax.scatter(x_axis[pre_finite], pre[pre_finite], s=8, color='red', alpha=0.7, label='Original')

            # 3) highlight newly filled points
            if np.any(filled):
                ax.scatter(x_axis[filled], post[filled], s=22, facecolor='limegreen',
                        edgecolor='black', linewidth=0.6, label='Filled')

            # small per-panel stats
            n_pre = np.count_nonzero(pre_finite)
            n_filled = np.count_nonzero(filled)
            pct = 100.0 * n_filled / max(1, (n_pre + n_filled))
            ax.set_title(f"Marker {m} — {colname}  (filled {n_filled}, {pct:.1f}%)")

            ax.set_ylabel("px")
            if m == n_markers - 1:
                ax.set_xlabel("Frame")

    # one legend for the whole figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncols=3)
    fig.suptitle(f"Cam {cam_idx}: Original points (red) vs Interpolated line (blue) — Filled samples in green", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.show()
    return confidence_thresholded_dlc_2d_array_XY


import numpy as np

import numpy as np
import pandas as pd
from pathlib import Path

def _interp_1d_with_guards(series: np.ndarray, *, max_gap: int | None = None) -> np.ndarray:
    """
    Interpolate a 1D array with NaNs.
    - If all NaN: return as-is.
    - If exactly one non-NaN: fill constant with that value.
    - Else: linear interpolate across NaNs.
    - If max_gap is set, any NaN run longer than max_gap remains NaN.
    """
    out = series.copy().astype(float)
    n = out.shape[0]
    idx = np.arange(n)
    valid = ~np.isnan(out)

    # Case 1: no valid samples
    if not valid.any():
        return out  # keep all-NaN

    # Case 2: single valid sample
    if valid.sum() == 1:
        out[:] = out[valid][0]
        return out

    # Linear interpolation
    out[~valid] = np.interp(idx[~valid], idx[valid], out[valid])

    # Respect max_gap if provided: reinsert NaNs for long gaps
    if max_gap is not None and max_gap >= 0:
        # find contiguous NaN runs in original series
        isn = ~valid
        if isn.any():
            # pad with False at ends to catch edge runs
            padded = np.r_[False, isn, False]
            changes = np.flatnonzero(padded[1:] != padded[:-1])
            # pairs (start, end) of NaN runs in [start, end)
            runs = list(zip(changes[0::2], changes[1::2]))
            for start, end in runs:
                if (end - start) > max_gap:
                    out[start:end] = np.nan

    return out


def linear_interpolate_2d(array_xy: np.ndarray, *, max_gap: int | None = None) -> np.ndarray:
    """
    Interpolate along frames for (n_cams, n_frames, n_markers, 2).
    Only X/Y channels are interpolated.
    """
    n_cams, n_frames, n_markers, two = array_xy.shape
    assert two == 2, "Expect last dim = 2 for X,Y"
    out = array_xy.copy().astype(float)

    for cam in range(n_cams):
        for m in range(n_markers):
            for dim in range(2):  # X then Y
                out[cam, :, m, dim] = _interp_1d_with_guards(
                    out[cam, :, m, dim], max_gap=max_gap
                )
    return out

def apply_confidence_threshold(array, threshold):
    """
    Set X,Y values to NaN where the corresponding confidence value is below threshold.
    """
    mask = array[..., 2] < threshold  # Shape: (num_cams, num_frames, num_markers)
    array[mask, 0] = np.nan  # Set X to NaN where confidence is low
    array[mask, 1] = np.nan  # Set Y to NaN where confidence is low
    return array



if __name__ == '__main__':

    path_to_folder_with_dlc_csvs = Path(r'D:\sfn\michael_wobble\recording_12_07_09_gmt-5__MDN_wobble_3\dlc_data')
    dlc_2d_array = compile_dlc_csvs(path_to_folder_with_dlc_csvs,
                                    confidence_threshold=.5)


    f = 2