from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, HTTPException, Request, BackgroundTasks, Form, File
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, Response
from starlette.middleware.cors import CORSMiddleware
import numpy as np
from pathlib import Path
import logging
import cv2
from io import BytesIO
from fastapi.encoders import jsonable_encoder
import base64


import logging
from tqdm import tqdm
from skellymodels.create_model_skeleton import create_mediapipe_skeleton_model, create_openpose_skeleton_model, create_qualisys_skeleton_model, create_rigid_mediapipe_skeleton_model
from skellymodels.model_info.mediapipe_model_info import MediapipeModelInfo

from multiprocessing import Pool
import time
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# recording_folder_path = Path(r'C:\Users\aaron\FreeMocap_Data\recording_sessions\freemocap_test_data')
# recording_folder_path = Path(r'D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_13_48_44_MDN_treadmill_2')
# recording_folder_path = Path(r'C:\Users\aaron\FreeMocap_Data\recording_sessions\sesh_2022-09-19_16_16_50_in_class_jsm')
# recording_folder_path = Path(r'D:\2024-04-25_P01\1.0_recordings\sesh_2024-04-25_15_44_19_P01_WalkRun_Trial1')
recording_folder_path = Path(r'D:\2024-04-25_P01\1.0_recordings\sesh_2024-04-25_15_55_43_P01_WalkRun_Trial2')
# recording_folder_path = Path(r'D:\philip_slackline')

# recording_folder_path = Path(r'D:\2024-08-01_treadmill_KK_JSM_ATC\1.0_recordings\sesh_2024-08-01_15_51_10_KK_treadmill_2')
# recording_folder_path = Path(r'D:\sfn\michael_wobble\recording_12_07_09_gmt-5__MDN_wobble_3')
# recording_folder_path = Path(r'D:\2023-06-14_rowing\1.0_recordings\sesh_2023-06-14_17_15_38_rowing_ATC_0')
# recording_folder_path = Path(r'D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_15_36_03_MDN_OneLeg_Trial1') #no need for centered data

output_data_folder_path = recording_folder_path / 'output_data'
# mediapipe_output_data_folder_path = recording_folder_path / 'output_data'/'rigid_bones_data'
mediapipe_output_data_folder_path = recording_folder_path / 'output_data'/'component_mediapipe_depth_pro_side'
# tracker_type = 'mediapipe'
# mediapipe_output_data_folder_path = recording_folder_path / 'output_data'/'origin_aligned_data'
qualisys_output_data_folder_path = recording_folder_path / 'output_data'
# tracker_type = 'mediapipe'
# data_3d_path = output_data_folder_path / f'{tracker_type}_body_3d_xyz.npy'

# recording_folder_path = Path(r'D:\mdn_treadmill_for_testing') #no need for centered data
# mediapipe_output_data_folder_path = recording_folder_path / 'output_data'


video_name = recording_folder_path/'test_video.mp4'
annotated_video_folder_path = recording_folder_path/'synchronized_videos'

list_of_annotated_videos = list(annotated_video_folder_path.glob('*.mp4'))

# Global variable to store frames
frames = {}
results_dict = None

@asynccontextmanager
async def lifespan_manager(app:FastAPI):
    logger.info("Starting up FastAPI app - access API backend interface at http://localhost:8000/docs")
    global results_dict
    results_dict = preproccess_annotated_videos(list_of_annotated_videos)
    yield
    logger.info("Shutting down FastAPI app")

app = FastAPI(lifespan=lifespan_manager)

origins = ["http://localhost:5173"]  
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/data/{tracker_type}")
async def get_data(tracker_type:str):
    try:
        if tracker_type == 'mediapipe':
            data3d = np.load(mediapipe_output_data_folder_path / 'mediapipe_depth_pro_body_3d_xyz.npy')
            data3d = data3d[:,0:33,:]
            # skeleton = create_rigid_mediapipe_skeleton_model()
            skeleton = create_mediapipe_skeleton_model()
        elif tracker_type == 'qualisys':
            data3d = np.load(qualisys_output_data_folder_path / 'mediapipe_body_3d_xyz.npy')
            skeleton = create_rigid_mediapipe_skeleton_model()
        else:
            raise HTTPException(status_code=400, detail="Unknown tracker type")
        
        skeleton.integrate_freemocap_3d_data(data3d)
        return skeleton.to_custom_dict()
    except Exception as e:
        logger.error(f"Error serving data: {e}")
        raise HTTPException(status_code=500, detail=f"Error serving data: {e}")


# app.mount("/static", StaticFiles(directory="skeleton-visualization/fast_api"), name="static")

def preproccess_annotated_videos(list_of_video_paths:list[Path]):
    with Pool(processes=8) as pool:
        results = pool.map(capture_all_frames_from_video, list_of_video_paths)
    
    global results_dict    
    results_dict = {video_number: frames for video_number, frames in enumerate(results)}

    return results_dict

    f = 2

def capture_all_frames_from_video(path_to_video:Path):
    preprocessed_frames = []
    cap = cv2.VideoCapture(str(path_to_video))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_number in tqdm(range(total_frames), desc="Capturing frames"):
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"Error reading frame {frame_number}")
            continue

        new_height,new_width = int(frame.shape[0]/4), int(frame.shape[1]/4)
        frame = cv2.resize(frame, (new_width, new_height))

        ret, buffer = cv2.imencode('.webp', frame, [int(cv2.IMWRITE_WEBP_QUALITY), 70])
        if not ret:
            raise HTTPException(status_code=500, detail=f"Error encoding frame {frame_number}")

        preprocessed_frames.append(buffer.tobytes())
    cap.release()
    logger.info(f"Captured {len(preprocessed_frames)} frames")
    return preprocessed_frames

@app.get("/video-info")
async def get_video_info():
    return {
        "videos": [
            {
                "name": video_path.name,
                "frame_count": len(frames)
            }
            for video_path, frames in zip(list_of_annotated_videos, results_dict.values())
        ],
        "total_videos": len(results_dict)
    }



@app.get("/video/frames/{frame_index}")
async def get_video_frames(frame_index: int):
    global results_dict
    
    if results_dict is None:
        raise HTTPException(status_code=500, detail="Video data not initialized")
    
    frames = {}
    for video_id, video_frames in results_dict.items():
        if 0 <= frame_index < len(video_frames):
            # Convert bytes to base64-encoded string
            frames[video_id] = base64.b64encode(video_frames[frame_index]).decode('utf-8')
    
    return JSONResponse(content=frames)

@app.post("/upload-frames")
async def upload_frames(request: Request, background_tasks: BackgroundTasks):
    global frames
    try:
        start_time = time.time()
        form = await request.form()
        files = form.getlist("files")
        width = int(form.get("width", 0))
        height = int(form.get("height", 0))
        batch_index = int(form.get("batchIndex", 0))
        total_frames = int(form.get("totalFrames", 0))

        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")
        if width == 0 or height == 0:
            raise HTTPException(status_code=400, detail="Invalid width or height")

        logger.info(f"Received batch {batch_index} with {len(files)} files. Total frames: {total_frames}")

        for i, file in enumerate(files):
            frame_number = batch_index * 500 + i  
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            frames[frame_number] = img
        
        frames_list = list(frames.values())
        logger.info(f"Processed batch {batch_index} in {time.time() - start_time:.2f} seconds")
        logger.info(f"Received {len(frames)} frames out of {total_frames} expected.")

        if len(frames) >= total_frames:
            logger.info("All frames received. Starting video creation.")
            background_tasks.add_task(create_multi_video_composite, video_name, frames, results_dict, width, height)
            # background_tasks.add_task(create_composite_video, video_name, frames_list, preprocessed_frames, width, height)
            # background_tasks.add_task(create_combined_video, video_name, frames_list, preprocessed_frames, width, height)
            # background_tasks.add_task(create_video_from_frames, video_name, total_frames, width, height)
            return JSONResponse(status_code=202, content={'status': 'processing', 'message': 'Video creation started'})
        else:
            return JSONResponse(status_code=200, content={'status': 'success', 'message': f'Batch {batch_index} received'})

    except Exception as e:
        logger.error(f"Error in upload_frames: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
# def create_multi_video_composite(video_name, threejs_frames, video_frames_dict, width, height):
#     try:
#         # Get the size of the threejs frames
#         first_threejs_frame = list(threejs_frames.values())[0]
#         frame_height, frame_width = first_threejs_frame.shape[:2]

#         # Initialize VideoWriter
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         out = cv2.VideoWriter(str(video_name), fourcc, 30.0, (frame_width, frame_height))

#         # Calculate the size for each video overlay
#         num_videos = len(video_frames_dict)
#         overlay_height = int(frame_height / 4)
#         max_overlay_width = int(frame_width / 2)
#         padding = 5

#         # Pre-calculate video sizes
#         video_sizes = []
#         for video_frames in video_frames_dict.values():
#             if len(video_frames) > 0:
#                 first_frame = cv2.imdecode(np.frombuffer(video_frames[0], np.uint8), cv2.IMREAD_COLOR)
#                 aspect_ratio = first_frame.shape[1] / first_frame.shape[0]
#                 new_height = overlay_height
#                 new_width = min(int(new_height * aspect_ratio), max_overlay_width)
#                 video_sizes.append((new_width, new_height))
#             else:
#                 video_sizes.append((0, 0))

#         # Calculate the width for each column (including padding)
#         column_width = max(size[0] for size in video_sizes) + padding

#         total_frames = len(threejs_frames)
#         for frame_number in tqdm(range(total_frames), desc="Creating composite video"):
#             # Get the threejs frame
#             threejs_frame = cv2.resize(threejs_frames[frame_number], (frame_width, frame_height))
#             composite_frame = threejs_frame.copy()

#             # Prepare and overlay video frames
#             for i, (video_frames, (new_width, new_height)) in enumerate(zip(video_frames_dict.values(), video_sizes)):
#                 if frame_number < len(video_frames):
#                     video_frame = cv2.imdecode(np.frombuffer(video_frames[frame_number], np.uint8), cv2.IMREAD_COLOR)
#                     video_frame_resized = cv2.resize(video_frame, (new_width, new_height))

#                     # Calculate position for 2-column layout
#                     row = i // 2
#                     col = i % 2
#                     y_start = row * (overlay_height + padding)
#                     x_start = col * column_width

#                     # Overlay the video frame
#                     composite_frame[y_start:y_start+new_height, x_start:x_start+new_width] = video_frame_resized

#             out.write(composite_frame)

#         out.release()
#         logger.info(f"Composite video saved as {video_name}")

#     except Exception as e:
#         logger.error(f"Error in create_multi_video_composite: {str(e)}")


def create_multi_video_composite(video_name, threejs_frames, video_frames_dict, width, height):
    try:
        # Get the size of the threejs frames
        first_threejs_frame = list(threejs_frames.values())[0]
        frame_height, frame_width = first_threejs_frame.shape[:2]

        # Initialize VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_name), fourcc, 30.0, (frame_width, frame_height))

        # Calculate the size for each video overlay
        num_videos = len(video_frames_dict)
        overlay_height = int(frame_height / 4)
        max_overlay_width = int(frame_width / 2)
        padding = 5

        # Pre-calculate video sizes
        video_sizes = []
        for video_frames in video_frames_dict.values():
            if len(video_frames) > 0:
                first_frame = cv2.imdecode(np.frombuffer(video_frames[0], np.uint8), cv2.IMREAD_COLOR)
                aspect_ratio = first_frame.shape[1] / first_frame.shape[0]
                new_height = overlay_height
                new_width = min(int(new_height * aspect_ratio), max_overlay_width)
                video_sizes.append((new_width, new_height))
            else:
                video_sizes.append((0, 0))

        # Calculate the width for each column (including padding)
        column_width = max(size[0] for size in video_sizes) + padding

        total_frames = len(threejs_frames)
        for frame_number in tqdm(range(total_frames), desc="Creating composite video"):
            # Get the threejs frame
            threejs_frame = cv2.resize(threejs_frames[frame_number], (frame_width, frame_height))
            composite_frame = threejs_frame.copy()

            # Prepare and overlay video frames
            for i, (video_frames, (new_width, new_height)) in enumerate(zip(video_frames_dict.values(), video_sizes)):
                if frame_number < len(video_frames):
                    video_frame = cv2.imdecode(np.frombuffer(video_frames[frame_number], np.uint8), cv2.IMREAD_COLOR)
                    video_frame_resized = cv2.resize(video_frame, (new_width, new_height))

                    # Calculate position for 2-column layout
                    row = i // 2
                    col = i % 2
                    y_start = row * (overlay_height + padding)
                    x_start = col * column_width

                    # Ensure the overlay fits within composite frame dimensions
                    target_area = composite_frame[y_start:y_start+new_height, x_start:x_start+new_width]
                    if target_area.shape[:2] != video_frame_resized.shape[:2]:
                        logger.warning(f"Frame size mismatch at frame {frame_number}, adjusting size.")
                        new_height, new_width = target_area.shape[:2]
                        video_frame_resized = cv2.resize(video_frame_resized, (new_width, new_height))

                    # Overlay the video frame
                    composite_frame[y_start:y_start+new_height, x_start:x_start+new_width] = video_frame_resized

            out.write(composite_frame)

        out.release()
        logger.info(f"Composite video saved as {video_name}")

    except Exception as e:
        logger.error(f"Error in create_multi_video_composite: {str(e)}")

def create_combined_video(video_name, threejs_frames, video_frames, width, height):
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_name), fourcc, 30.0, (width * 2, height))

        num_frames = min(len(threejs_frames), len(video_frames))
        for i in tqdm(range(num_frames)):
            threejs_frame = threejs_frames[i]
            video_frame = video_frames[i]
            
            video_img = cv2.imdecode(np.frombuffer(video_frame, np.uint8), cv2.IMREAD_COLOR)
            video_img = cv2.resize(video_img, (width, height))
            threejs_frame = cv2.resize(threejs_frame, (width, height))
            
            combined_frame = np.hstack((video_img, threejs_frame))
            out.write(combined_frame)

        out.release()
        logger.info(f"Combined video saved as {video_name}")

    except Exception as e:
        logger.error(f"Error in create_combined_video: {str(e)}")

def create_composite_video(video_name, threejs_frames, video_frames, width, height):
    try:
        # Get the size of the threejs frames (height, width, channels)
        first_threejs_frame = threejs_frames[0]
        frame_height, frame_width = first_threejs_frame.shape[:2]

        # Initialize VideoWriter with the correct frame size (width, height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_name), fourcc, 30.0, (frame_width, frame_height))

        # Get the size of the first video frame to determine aspect ratio
        first_video_frame = cv2.imdecode(np.frombuffer(video_frames[0], np.uint8), cv2.IMREAD_COLOR)
        video_height, video_width = first_video_frame.shape[:2]
        video_aspect_ratio = video_width / video_height

        # Calculate the size of the overlay video
        overlay_height = int(frame_height / 4)
        overlay_width = int(overlay_height * video_aspect_ratio)

        # Process frames
        num_frames = min(len(threejs_frames), len(video_frames))
        for i in tqdm(range(num_frames)):
            # Resize and copy the threejs frame
            threejs_frame = cv2.resize(threejs_frames[i], (frame_width, frame_height))
            composite_frame = threejs_frame.copy()

            # Decode and resize the video frame
            video_frame = cv2.imdecode(np.frombuffer(video_frames[i], np.uint8), cv2.IMREAD_COLOR)
            video_frame_resized = cv2.resize(video_frame, (overlay_width, overlay_height))

            # Overlay the video frame onto the threejs frame
            composite_frame[0:overlay_height, 0:overlay_width] = video_frame_resized

            # Write the composite frame to the video
            out.write(composite_frame)

        out.release()
        logger.info(f"Composite video saved as {video_name}")

    except Exception as e:
        logger.error(f"Error in create_composite_video: {str(e)}")



@app.get("/")
async def get_index():
    logger.info("Serving index.html")
    return FileResponse("backend/static/index.html")


@app.get("/video/frame/{frame_index}")
async def get_video_frame(frame_index:int):
    if frame_index < 0 or frame_index >= len(preprocessed_frames):
        raise HTTPException(status_code=404, detail="Frame not found")
    return StreamingResponse(BytesIO(preprocessed_frames[frame_index]), media_type="image/webp")



@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"Message text was: {data}")

def create_video_from_frames(output_filename, total_frames, width, height):
    global frames
    try:
        start_time = time.time()
        logger.info(f"Starting video creation with {total_frames} frames")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_filename), fourcc, 30.0, (width, height))

        for i in tqdm(range(total_frames), desc="Creating video"):
            frame = frames.get(i)
            if frame is not None:
                out.write(frame)
            else:
                logger.warning(f"Missing frame: {i}")

        out.release()
        logger.info(f"Video saved as {output_filename}")
        logger.info(f"Video creation completed in {time.time() - start_time:.2f} seconds")

        # Clear the frames from memory
        frames.clear()
        logger.info("Cleared frames from memory")

    except Exception as e:
        logger.error(f"Error in create_video_from_frames: {str(e)}")
    finally:
        # Ensure frames are cleared even if an error occurred
        frames.clear()

# @app.get("/video/frame/{frame_index}")
# async def get_frame(frame_index: int):
#     with cap_lock:
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         logging.info(f"Total frames: {total_frames}")

#         if frame_index < 0 or frame_index >= total_frames:
#             raise HTTPException(status_code=404, detail="Frame not found")

#         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
#         logging.info(f"Reading frame {frame_index}")
#         ret, frame = cap.read()

#     if not ret:
#         raise HTTPException(status_code=500, detail="Error reading frame")

#     # Resize the frame to a lower resolution
#     new_height,new_width = int(frame.shape[0]/4), int(frame.shape[1]/4)
#     frame = cv2.resize(frame, (new_width, new_height))

#     # Adjust JPEG quality to reduce size
#     ret, buffer = cv2.imencode('.webp', frame, [int(cv2.IMWRITE_WEBP_QUALITY), 70])
#     if not ret:
#         raise HTTPException(status_code=500, detail="Error encoding frame")

#     io_buf = BytesIO(buffer.tobytes())
#     return StreamingResponse(io_buf, media_type="image/webp")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")