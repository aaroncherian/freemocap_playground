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
from skellymodels.create_model_skeleton import create_mediapipe_skeleton_model, create_openpose_skeleton_model
from pathlib import Path
# HTTP Server
recording_folder_path = Path(r'D:\steen_pantsOn_gait_3_cameras')
# recording_folder_path = Path(r'D:\2023-06-07_TF01\1.0_recordings\treadmill_calib\sesh_2023-06-07_11_55_05_TF01_flexion_neg_5_6_trial_1')
output_data_folder_path = recording_folder_path / 'output_data'
# data_3d_path = output_data_folder_path / 'openpose_body_3d_xyz.npy'
data_3d_path = output_data_folder_path / 'mediapipe_body_3d_xyz.npy'
# ik_results_path = output_data_folder_path / 'IK_results.mot'

class HttpHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        print(f"Requested path: {self.path}")
        if self.path == '/':
            self.path = '/skeleton-visualization/index.html'
        elif self.path.startswith('/data'):
            self.serve_data()
            return
        elif self.path.startswith('/trajectory_data'):
            self.serve_trajectory_data()
            return
        else:
            self.path = '/skeleton-visualization/' + self.path.lstrip('/')
        return SimpleHTTPRequestHandler.do_GET(self)

    def serve_data(self):
        try:
            np_data = np.load(data_3d_path)
            
            # mediapipe_skeleton = create_openpose_skeleton_model()
            mediapipe_skeleton = create_mediapipe_skeleton_model()
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

    def serve_trajectory_data(self):
        try:
            query = self.path.split('?')[-1]
            params = dict(qc.split("=") for qc in query.split("&"))
            joint = params.get('joint', 'right_ankle')  # default to 'right_ankle' if no joint is specified

            np_data = np.load(data_3d_path)

            # mediapipe_skeleton = create_openpose_skeleton_model()
            mediapipe_skeleton = create_mediapipe_skeleton_model()
            mediapipe_skeleton.integrate_freemocap_3d_data(np_data)

            data = mediapipe_skeleton.trajectories[joint]
            data_x = data[:, 0].tolist()
            data_y = data[:, 1].tolist()
            data_z = data[:, 2].tolist()

            trajectory_data = {
                'x': data_x,
                'y': data_y,
                'z': data_z
            }

            response = json.dumps(trajectory_data)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(response.encode())
            print(f"Trajectory data served successfully for joint: {joint}")

        except Exception as e:
            print(f"Error serving trajectory data: {e}")
            self.send_response(500)
            self.end_headers()

    # def serve_ankle_angle_data(self):
    #     try:

    #         ik_data =  pd.read_csv(ik_results_path, sep='\t', skiprows=10)
    #         right_ankle_angle = ik_data['ankle_angle_r'].tolist()

    #         response = json.dumps(right_ankle_angle)
    #         self.send_response(200)
    #         self.send_header('Content-type', 'application/json')
    #         self.end_headers()
    #         self.wfile.write(response.encode())
    #         print("Ankle angle data served successfully")

        # except Exception as e:
        #     print(f"Error serving ankle angle data: {e}")
        #     self.send_response(500)
        #     self.end_headers()



            





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
    templates_path = './skeleton-visualization/'
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
