from pathlib import Path
import cv2
from tqdm import tqdm
import logging

import mediapipe as mp
import numpy as np
import depth_pro
import torch


def process_image(image, transform):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Remove alpha channel if present
    if rgb_image.shape[2] == 4:
        rgb_image = rgb_image[:, :, :3]
    results = pose_estimator.process(rgb_image)

    image = transform(rgb_image)
    prediction = model.infer(image, f_px=None)
    depth_array = prediction['depth'].cpu().numpy()
    pose_data.append(results)
    depth_data.append(depth_array)

def process_recorded_data(
    pose_data: list, width: int, height: int, depth_data: list, num_landmarks: int = 33
):
    processed_data_array = []

    for count, result in tqdm(enumerate(pose_data), desc='Processing Pose Data'):
        if result.pose_landmarks:  # Check if landmarks exist
            frame_data = []
            for landmark_data in result.pose_landmarks.landmark:
                x = landmark_data.x * width
                y = landmark_data.y * height
                # Handle out-of-bounds landmarks
                if not (0 <= landmark_data.x <= 1 and 0 <= landmark_data.y <= 1):
                    print(f"Out of bounds in frame {count}: x={landmark_data.x}, y={landmark_data.y}")
                    z = np.nan  # Set to NaN
                else:
                    z = depth_data[count][int(y), int(x)] * 1000  # Convert to mm
                frame_data.append([x, y, z])
        else:
            # Fill with NaN for missing landmarks
            frame_data = [[np.nan, np.nan, np.nan]] * num_landmarks
        
        processed_data_array.append(frame_data)

    return np.array(processed_data_array)

logger = logging.getLogger(__name__)
mp_pose = mp.solutions.pose

pose_estimator = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    min_detection_confidence=0.5,
)


input_video_filepath = Path(r"D:\2024-04-25_P01\1.0_recordings\sesh_2024-04-25_15_55_43_P01_WalkRun_Trial2\synchronized_videos\sesh_2024-04-25_15_55_43_P01_WalkRun_Trial2_synced_Cam6.mp4")

cap = cv2.VideoCapture(str(input_video_filepath))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

pose_data = []
depth_data = []

model, transform = depth_pro.create_model_and_transforms(device=torch.device("cuda"),
                                                         precision=torch.float16 )
model.eval()


logging.info('Creating Depth Model')
iterator = tqdm(
    range(number_of_frames),
    desc=f"processing video: {Path(input_video_filepath).name}",
    total=number_of_frames,
    colour="magenta",
    unit="frames",
    dynamic_ncols=True,
)

ret,frame = cap.read()

for frame_number in iterator:
    if not ret or frame is None:
        logger.error(
                    f"Failed to load an image from: {str(input_video_filepath)} for frame_number: {frame_number}"
                )
        raise ValueError(f"Failed to load an image from: {str(input_video_filepath)} for frame_number: {frame_number}")
    process_image(frame, transform)
    ret, frame = cap.read()
cap.release()

mediapipe_with_depth_array = process_recorded_data(pose_data, width, height, depth_data)
np.save(r'D:\2024-04-25_P01\1.0_recordings\sesh_2024-04-25_15_55_43_P01_WalkRun_Trial2\output_data\component_mediapipe_depth_pro_side\mediapipe_depth_pro_body_3d_xyz.npy', mediapipe_with_depth_array)
f = 2





    # with mp_pose.Pose(
    # static_image_mode=False,
    # model_complexity=2,
    # min_detection_confidence=0.5,
    # ) as pose:
    #     results = pose.process(cv2.cvtColor(image_frame, cv2.COLOR_BGR2RGB))
    #     img_height, img_width, _ = image_frame.shape

    #     mediapipe_array = []

    #     for landmark_data in results.pose_landmarks.landmark:
    #         # mediapipe_array.append([landmark_data.x*img.shape[0], landmark_data.y*img.shape[1]])
    #         x = landmark_data.x * img_width
    #         y = landmark_data.y * img_height
    #         z = landmark_data.z * img_width  # Z is relative to the width10
    #         mediapipe_array.append([x, y, z])
