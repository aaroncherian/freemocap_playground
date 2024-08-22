import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
from pathlib import Path
import logging

from skellymodels.create_model_skeleton import create_mediapipe_skeleton_model, create_openpose_skeleton_model
from skellymodels.model_info.mediapipe_model_info import MediapipeModelInfo

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

recording_folder_path = Path(r'D:\2023-06-07_TF01\1.0_recordings\treadmill_calib\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1')
output_data_folder_path = recording_folder_path / 'output_data'
tracker_type = 'mediapipe'
data_3d_path = output_data_folder_path / f'{tracker_type}_body_3d_xyz.npy'
ik_results_path = output_data_folder_path / 'IK_results.mot'

joint_to_angle_mapping = {
    'right_hip': 'hip_flexion_r',
    'left_hip': 'hip_flexion_l',
    'right_knee': 'knee_angle_r',
    'left_knee': 'knee_angle_l',
    'right_ankle': 'ankle_angle_r',
    'left_ankle': 'ankle_angle_l'
}

@asynccontextmanager
async def lifespan_manager(app:FastAPI):
    logger.info("Starting up FastAPI app - access API backend interface at http://localhost:8000/docs")
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

# app.mount("/static", StaticFiles(directory="skeleton-visualization/fast_api"), name="static")

@app.get("/")
async def get_index():
    logger.info("Serving index.html")
    return FileResponse("skeleton-visualization/fast_api/index.html")

@app.get("/available_joint_names")
async def get_available_joint_names():
    try:
        joint_names = list(joint_to_angle_mapping.keys())
        return {"joint_names":joint_names}
    except Exception as e:
        logger.error(f"Error serving data: {e}")
        raise HTTPException(status_code=500, detail=f"Error serving data: {e}")
    
@app.get("/data")
async def get_data():
    try:
        np_data = np.load(data_3d_path)

        if tracker_type == 'mediapipe':
            skeleton = create_mediapipe_skeleton_model()
        else:
            raise HTTPException(status_code=400, detail="Unknown tracker type")
        
        skeleton.integrate_freemocap_3d_data(np_data)
        return skeleton.to_custom_dict()
    except Exception as e:
        logger.error(f"Error serving data: {e}")
        raise HTTPException(status_code=500, detail=f"Error serving data: {e}")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"Message text was: {data}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")