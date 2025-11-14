from nicegui import ui
from skellymodels.managers.human import Human

class ThreeJSPlot:
    def __init__(self, human:Human):
        self.marker_data_unscaled = human.body.xyz.as_array()
        self.spheres = {}
        self.scene = None

    def get_frame_data(self, frame_number:int):
        return self.marker_data_unscaled[frame_number,:,:]/100

    def create_scene(self):
        with ui.scene().classes('w-[800px] h-[800px]') as scene:
            data = self.get_frame_data(0) 
            for i, marker in enumerate(data):
                self.spheres[i] = scene.sphere(radius=0.2).material('#4488ff').move(*marker)
        self.scene = scene
        return scene

    def update_scene(self, frame_number:int):
        new_data = self.get_frame_data(frame_number)
        for i, marker in enumerate(new_data):
            self.spheres[i].move(*marker)  
