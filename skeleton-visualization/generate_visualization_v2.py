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

from skellymodels.model_info.mediapipe_model_info import MediapipeModelInfo
# HTTP Server
class HttpHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        print(f"Requested path: {self.path}")
        if self.path == '/':
            self.path = '/skeleton-visualization/index.html'
        elif self.path == '/data':
            self.serve_data()
            return
        elif self.path == '/trajectory_data':
            self.serve_trajectory_data()
            return
        else:
            self.path = '/skeleton-visualization/' + self.path.lstrip('/')
        return SimpleHTTPRequestHandler.do_GET(self)

    def serve_data(self):
        try:
            path_to_data = r"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_13_37_32_MDN_treadmill_1\output_data\mediapipe_body_3d_xyz.npy"
            np_data = np.load(path_to_data)

            # Reshape and prepare the data for JSON response
            num_frames, num_markers, _ = np_data.shape
            data = [[np_data[frame, marker].tolist() for marker in range(num_markers)] for frame in range(num_frames)]

            response = json.dumps(data)
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
            path_to_data = r"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_13_37_32_MDN_treadmill_1\output_data\mediapipe_body_3d_xyz.npy"
            np_data = np.load(path_to_data)

            mediapipe_indices = MediapipeModelInfo.body_landmark_names
            right_ankle_index = mediapipe_indices.index('right_ankle')
            right_ankle_data = np_data[:, right_ankle_index, 0].tolist()  # Extracting the x-coordinate data
            
            response = json.dumps(right_ankle_data)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(response.encode())
            print("Trajectory data served successfully")

        except Exception as e:
            print(f"Error serving trajectory data: {e}")
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
