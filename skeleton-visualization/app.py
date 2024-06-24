from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO
import numpy as np
import json
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
import time

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def data():
    path_to_data = r"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_13_37_32_MDN_treadmill_1\output_data\mediapipe_body_3d_xyz.npy"
    np_data = np.load(path_to_data)

    # Reshape and prepare the data for JSON response
    num_frames, num_markers, _ = np_data.shape
    data = [[np_data[frame, marker].tolist() for marker in range(num_markers)] for frame in range(num_frames)]

    # data = [np.random.rand(num_markers, 3).tolist() for _ in range(num_frames)]
    # return data

    return jsonify(data)

@socketio.on('connect')
def handle_connect():
    # Notify the client that the connection is established
    socketio.emit('connected')

# Watchdog event handler
class WatchdogHandler(FileSystemEventHandler):
    def __init__(self, socketio):
        self.socketio = socketio

    def on_modified(self, event):
        if event.src_path.endswith(".html"):
            print(f'{event.src_path} has been modified')
            self.socketio.emit('reload')

def start_watchdog(socketio):
    event_handler = WatchdogHandler(socketio)
    observer = Observer()
    observer.schedule(event_handler, path='.', recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == '__main__':
    # Start the watchdog in a separate thread

    path_to_data = r"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_13_37_32_MDN_treadmill_1\output_data\mediapipe_body_3d_xyz.npy"
    data = np.load(path_to_data)

    watchdog_thread = threading.Thread(target=start_watchdog, args=(socketio,))
    watchdog_thread.daemon = True
    watchdog_thread.start()

    socketio.run(app, debug=False)
