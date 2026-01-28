"""
Run EasyViTPose on a video and save an annotated video.

Notes:
- VitInference expects RGB images for inference().
- draw() returns an RGB image with keypoints drawn (it uses cached state from inference()).
- OpenCV VideoWriter expects BGR frames.

Tip:
- Use is_video=True to enable SORT tracking (consistent IDs across frames).
- Call model.reset() before starting a new video.
"""

from __future__ import annotations

from pathlib import Path
import cv2
from easy_ViTPose import VitInference


def annotate_video(
    input_video_path: str | Path,
    output_video_path: str | Path,
    model_path: str | Path,
    yolo_path: str | Path,
    model_name: str = "b",
    yolo_size: int = 640,
    device: str | None = None,
    show_yolo: bool = False,
    show_raw_yolo: bool = False,
    confidence_threshold: float = 0.5,
    yolo_step: int = 1,
    single_pose: bool = False,
    max_frames: int | None = None,
) -> None:
    input_video_path = Path(input_video_path)
    output_video_path = Path(output_video_path)
    model_path = str(model_path)
    yolo_path = str(yolo_path)

    if not input_video_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_video_path}")

    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0  # fallback

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Choose a codec. 'mp4v' is widely available; 'avc1' may work if H.264 codecs are installed.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(
            f"Could not open VideoWriter for: {output_video_path}\n"
            f"Try changing codec (e.g., 'avc1') or output extension."
        )

    # Create model in "video" mode for tracking
    model = VitInference(
        model=str(model_path),
        yolo=str(yolo_path),
        model_name=model_name,      # required for .pth
        yolo_size=yolo_size,
        is_video=True,
        device=device,
        yolo_step=yolo_step,
        single_pose=single_pose,
    )
    model.reset()

    frame_idx = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            # Convert to RGB for inference
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Run inference (updates internal state used by draw())
            _ = model.inference(frame_rgb)

            # Draw annotation (returns RGB)
            annotated_rgb = model.draw(
                show_yolo=show_yolo,
                show_raw_yolo=show_raw_yolo,
                confidence_threshold=confidence_threshold,
            )

            # Convert back to BGR for writing
            annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
            writer.write(annotated_bgr)

            frame_idx += 1
            if max_frames is not None and frame_idx >= max_frames:
                break

            if frame_idx % 50 == 0:
                print(f"Processed {frame_idx} frames...")

    finally:
        cap.release()
        writer.release()

    print(f"Saved annotated video to: {output_video_path}")


if __name__ == "__main__":
    annotate_video(
        input_video_path=r"D:\2025_07_31_JSM_pilot\freemocap\2025-07-31_16-35-10_GMT-4_jsm_treadmill_trial_1\synchronized_videos\2025-07-31_16-35-10_GMT-4_jsm_treadmill_trial_1.camera2.mp4",
        output_video_path=r"D:\2025_07_31_JSM_pilot\freemocap\2025-07-31_16-35-10_GMT-4_jsm_treadmill_trial_1\VIT_pose_sesh_2022-09-19_16_16_50_in_class_jsm_synced_Cam1.mp4",
        model_path=r"C:\Users\aaron\Documents\GitHub\easy_ViTPose\models\vitpose-b-wholebody.pth",
        yolo_path=r"C:\Users\aaron\Documents\GitHub\easy_ViTPose\models\yolov8l.pt",
        model_name="b",
        yolo_size=1024,          # you used 1000; 640–1000 are common. Higher = slower.
        device="cuda",             # None -> cuda/mps/cpu auto
        show_yolo=True,         # set True if you want bbox overlay
        confidence_threshold=0.5,
        yolo_step=1,             # run yolo every frame; increase for speed on videos
        single_pose=False,       # True if you know there's only one person (faster, no tracking)
        max_frames=2000,         # set e.g. 300 for quick test
    )

