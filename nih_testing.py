import numpy as np
import matplotlib.pyplot as plt


def plot_xy_per_camera(data: np.ndarray,
                       marker_idx: int,
                       frame_range: tuple[int, int]):
    """
    data shape: [camera, frame, marker, dimension]

    marker_idx : marker to plot
    frame_range: (start_frame, end_frame)
    """

    start, end = frame_range
    num_cameras = data.shape[0]

    frames = np.arange(start, end)

    fig, axes = plt.subplots(num_cameras, 2,
                             figsize=(10, 3 * num_cameras),
                             sharex=True)

    # Handle case where there is only one camera
    if num_cameras == 1:
        axes = axes.reshape(1, 2)

    for cam in range(num_cameras):

        x = data[cam, start:end, marker_idx, 0]
        y = data[cam, start:end, marker_idx, 1]

        axes[cam, 0].plot(frames, x, color="blue")
        axes[cam, 0].set_ylabel(f"Cam {cam}")
        axes[cam, 0].set_title("X")

        axes[cam, 1].plot(frames, y, color="red")
        axes[cam, 1].set_title("Y")

    axes[-1, 0].set_xlabel("Frame")
    axes[-1, 1].set_xlabel("Frame")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from pathlib import Path
    import numpy as np

    array = np.load(Path(r"D:\validation\data\2026_03_04_ML\2026-03-04_19-27-37_GMT-5_ml_nih_trial_2\output_data\raw_data\mediapipe_2dData_numCams_numFrames_numTrackedPoints_pixelXY.npy"))
    plot_xy_per_camera(
        data=array,
        marker_idx=24,
        frame_range=(7000, 8000)
    )