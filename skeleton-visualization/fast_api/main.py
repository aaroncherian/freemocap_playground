import asyncio
import websockets
import json
import numpy as np
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import webbrowser
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import pandas as pd
from skellymodels.model_info.mediapipe_model_info import MediapipeModelInfo
from skellymodels.create_model_skeleton import create_mediapipe_skeleton_model, create_openpose_skeleton_model
from pathlib import Path
# HTTP Server

# recording_folder_path = Path(r'D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_13_37_32_MDN_treadmill_1')
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

tracker_type = 'mediapipe'
# data_3d_path = output_data_folder_path / 'openpose_body_3d_xyz.npy'
data_3d_path = output_data_folder_path / f'{tracker_type}_body_3d_xyz.npy'



class HttpHandler(SimpleHTTPRequestHandler):

    def do_GET(self):
        print(f"Requested path: {self.path}")
        if self.path == '/':
            self.path = '/skeleton-visualization/fast_api/index.html'  # Ensure the correct path
        elif self.path == '/data':
            self.serve_data()
            return
        elif self.path == '/available_joint_names':
            self.serve_available_joint_names()
            return
        else:
            self.path = '/skeleton-visualization/fast_api/' + self.path.lstrip('/')  # Ensure the correct path
        return SimpleHTTPRequestHandler.do_GET(self)

    def serve_available_joint_names(self):
        try:
            joint_names = list(joint_to_angle_mapping.keys())

            response = json.dumps(joint_names)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(response.encode())
            print('Joint names served successfully')

        except Exception as e:
            print(f"Error serving data: {e}")
            self.send_response(500)
            self.end_headers()

    def serve_data(self):
        try:
            np_data = np.load(data_3d_path)
            
            if tracker_type == 'mediapipe':
                mediapipe_skeleton = create_mediapipe_skeleton_model()
            elif tracker_type == 'openpose':
                mediapipe_skeleton = create_openpose_skeleton_model()
            else:
                print('Unknown tracker type')
            mediapipe_skeleton.integrate_freemocap_3d_data(np_data)
            response = mediapipe_skeleton.to_json()

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(response.encode())
            print("Data served successfully")
        except Exception as e:
            print(f"Error serving data: {e}")
            self.send_response(500)
            self.end_headers()



# WebSocket handler to notify clients about file changes
connected_clients = set()
main_event_loop = asyncio.get_event_loop()

async def websocket_handler(websocket, path):
    connected_clients.add(websocket)
    try:
        async for message in websocket:
            pass
    finally:
        connected_clients.remove(websocket)

def notify_change():
    for client in connected_clients:
        asyncio.run_coroutine_threadsafe(client.send("reload"), main_event_loop)

# Watchdog event handler
class WatchdogHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith("index.html"):
            print(f'{event.src_path} has been modified')
            notify_change()

def start_watchdog():
    event_handler = WatchdogHandler()
    templates_path = './skeleton-visualization/fast_api/'
    observer = Observer()
    observer.schedule(event_handler, path=templates_path, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# Create and run the HTTP server
def run_http_server():
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, HttpHandler)
    print("HTTP server running on http://localhost:8000")
    httpd.serve_forever()

# Create and run the WebSocket server
async def run_websocket_server():
    server = await websockets.serve(websocket_handler, "localhost", 8001)
    print("WebSocket server running on ws://localhost:8001")
    await server.wait_closed()

# Main function to run both servers
def main():
    global main_event_loop
    main_event_loop = asyncio.get_event_loop()

    # Start HTTP server in a separate thread
    http_thread = threading.Thread(target=run_http_server)
    http_thread.start()

    # Start watchdog in a separate thread
    watchdog_thread = threading.Thread(target=start_watchdog)
    watchdog_thread.start()

    # Open the web browser
    webbrowser.open('http://localhost:8000')

    # Run WebSocket server in the main thread
    main_event_loop.run_until_complete(run_websocket_server())

if __name__ == "__main__":
    main()
