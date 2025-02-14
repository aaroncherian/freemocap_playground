from skellymodels.experimental.model_redo.managers.human import Human
from skellymodels.experimental.model_redo.tracker_info.model_info import ModelInfo

from back_angle_plot import BackAnglePlot
from threejsplot import ThreeJSPlot
from trunk_inclination_plot import TrunkInclinationPlot, TrunkRotationPlot

from pathlib import Path
import numpy as np
from nicegui import ui


def calculate_spherical_angles(human: Human):
    body_3d_xyz = human.body.trajectories['3d_xyz']
    spine_vector = body_3d_xyz.segment_data['spine']['proximal'] - body_3d_xyz.segment_data['spine']['distal']
    spine_vector_magnitude = np.linalg.norm(spine_vector, axis=1)
    spine_vector_azimuthal = np.arctan2(spine_vector[:, 1], spine_vector[:, 0])
    spine_vector_polar = np.arccos(spine_vector[:,2]/(spine_vector_magnitude + 1e-9))

    return spine_vector_azimuthal, spine_vector_polar, spine_vector_magnitude


path_to_recording = Path(r'D:\recording_15_26_59_gmt-4__kk_bad_form')
path_to_output_data = path_to_recording / 'output_data'/ 'mediapipe_body_3d_xyz.npy'

model_info = ModelInfo(config_path = Path(__file__).parent/'mediapipe_just_body.yaml')
human = Human.from_numpy_array(
    name = 'human', 
    model_info=model_info,
    tracked_points_numpy_array=np.load(path_to_output_data)
)
num_frames = human.body.trajectories['3d_xyz'].num_frames

spine_vector_azimuthal, spine_vector_polar, spine_vector_magnitude = calculate_spherical_angles(
    human=human
)


back_plot = BackAnglePlot(azimuthal=spine_vector_azimuthal, 
                         polar=spine_vector_polar,
                         vector_magnitude=spine_vector_magnitude)
marker_viz = ThreeJSPlot(human=human)
trunk_inclination_plot = TrunkInclinationPlot(polar_angle_data=spine_vector_polar)
trunk_rotation_plot = TrunkRotationPlot(azimuthal_angle_data=spine_vector_azimuthal)

with ui.row():
    marker_viz.create_scene()
    plot3d = ui.plotly(back_plot.create_figure()).classes('w-[800px] h-[800px]')
    with ui.column():
        inclination_plot2d = trunk_inclination_plot.create_plot()
        rotation_plot2d = trunk_rotation_plot.create_plot()


def update_label(event):
    frame_label.text = f'Frame: {int(event.value)}'

def on_slider_change(event):
    plot3d.update_figure(back_plot.update_plot(event.value))
    update_label(event)
    marker_viz.update_scene(event.value)
    trunk_inclination_plot.update_plot(event.value)
    trunk_rotation_plot.update_plot(event.value)

frame_label = ui.label('Frame: 0')
ui.slider(
    min=0, 
    max=num_frames-1,
    step=1,
    value=0,
    on_change=on_slider_change
)

ui.run()
f = 2