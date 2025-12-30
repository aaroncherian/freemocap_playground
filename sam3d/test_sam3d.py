import os
os.environ["PYOPENGL_PLATFORM"] = "windows"  # must be set before imports using OpenGL

from pathlib import Path
import sys
import cv2
import numpy as np
from tqdm import tqdm

parent_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, parent_dir)

from sam3d.utils import setup_sam_3d_body, visualize_2d_results
from sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer
from huggingface_hub import login
from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info


login()

parent_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, parent_dir)

path_to_video = Path(
    r"C:\Users\aaron\freemocap_data\recording_sessions\freemocap_test_data\synchronized_videos\sesh_2022-09-19_16_16_50_in_class_jsm_synced_Cam1.mp4"
)
cap = cv2.VideoCapture(str(path_to_video))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# output video path
out_video_path = path_to_video.with_name(path_to_video.stem + "_sam2d_overlay.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

estimator = setup_sam_3d_body(
    hf_repo_id="facebook/sam-3d-body-dinov3",
    fov_name=None,
    detector_name=None,   
    segmentor_name=None,   
)


visualizer = SkeletonVisualizer()
visualizer.set_pose_meta(mhr70_pose_info)

all_keypoints_2d = []

pbar = tqdm(total=total_frames, desc="Processing Frames", unit="frame")
while True:
    ret, frame = cap.read()
    if not ret:
        break  

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    outputs = estimator.process_one_image(img_rgb)

    if len(outputs) == 0:
        keypoints_2d = np.full((70, 2), np.nan, dtype=np.float32)
        annotated_frame = frame  
    else:
        keypoints_2d = np.asarray(outputs[0]["pred_keypoints_2d"], dtype=np.float32)


        vis_list = visualize_2d_results(frame, outputs, visualizer)

        if len(vis_list) > 0:
            annotated_frame = vis_list[0]  
        else:
            annotated_frame = frame

    all_keypoints_2d.append(keypoints_2d)

    writer.write(annotated_frame)

    pbar.update(1)

pbar.close()
cap.release()
writer.release()

# stack keypoints: [num_frames, 70, 2]
all_keypoints_2d = np.stack(all_keypoints_2d, axis=0)

print("Saved annotated video to:", out_video_path)
print("Keypoints array shape:", all_keypoints_2d.shape)
