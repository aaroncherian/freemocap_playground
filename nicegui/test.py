from pathlib import Path
import numpy as np
from nicegui import ui

path_to_data = Path(r"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_13_48_44_MDN_treadmill_2\output_data\aligned_data\mediapipe_body_3d_xyz.npy")

body_data = np.load(path_to_data)
num_frames = body_data.shape[0]

def update_frame_to_use(frame:int = 0):
    data = body_data[frame,:,:]/100
    return data




f = 2
spheres = []
with ui.scene().classes('w-[800px] h-[800px]') as scene:
    for obj in list(scene.objects.values()):
        obj.delete()
    data = update_frame_to_use() 
    for marker in data:
            sphere = scene.sphere(radius=.2).material('#4488ff').move(marker[0], marker[1], marker[2])
            spheres.append(sphere)

def update_spheres(frame):
     new_data = update_frame_to_use(frame)
     for i, marker in enumerate(new_data):
          spheres[i].move(*marker)


slider = ui.slider(min=0, max=num_frames-1, step=1, value=0) \
    .on_value_change(lambda e: update_spheres(int(e.value)))
ui.label().bind_text_from(slider, 'value')

ui.run()

