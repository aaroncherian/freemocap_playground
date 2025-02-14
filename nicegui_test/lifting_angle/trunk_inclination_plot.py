import numpy as np
from nicegui import ui
class TrunkInclinationPlot:
    def __init__(self, polar_angle_data: np.ndarray):
        self.polar_angle_data = np.degrees(polar_angle_data)
        self.num_frames = len(polar_angle_data)
        self.frames = np.arange(self.num_frames)
        self.plot = None
        self.vertical_line = None
            
    def create_plot(self):
        self.echart = ui.echart({
            'title': {
                'text': 'Trunk Inclination vs. Time'
            },
            'xAxis': {'type': 'category',
                      'name': 'Frame #',
                      'nameLocation': 'middle',
                      'nameGap': 25,
                      'nameTextStyle': {
                          'fontSize': 14 
                      }},
            'yAxis': {
                'type': 'value',
                'name': 'Trunk Inclination (degrees)',
                'nameLocation': 'middle',  
                'nameRotate': 90,  # Rotates the title
                'nameTextStyle': {
                    'padding': [0, 0, 20, 0], 
                    'fontSize': 14,  
                }},
            'series': [{
                'type': 'line',
                'data': list(self.polar_angle_data),
                'markLine': {  
                    'animation': False,
                    'data': [
                        {'xAxis': 0, 'lineStyle': {'color': 'black', 'width': 1}}
                    ]
                }
            }]
        })

        self.echart.style('width: 800px; height: 400px;')
        return self.echart
    
    def update_plot(self, frame_idx: int):
        self.echart.options['series'][0]['markLine']['data'][0]['xAxis'] = frame_idx
        self.echart.update()
        


class TrunkRotationPlot:
    def __init__(self, azimuthal_angle_data: np.ndarray):
        self.azimuthal_angle_data = np.degrees(azimuthal_angle_data)
        self.num_frames = len(azimuthal_angle_data)
        self.frames = np.arange(self.num_frames)
        self.plot = None
        self.vertical_line = None
            
    def create_plot(self):
        self.echart = ui.echart({
            'title': {
                'text': 'Trunk Rotation vs. Time'
            },
            'xAxis': {'type': 'category',
                      'name': 'Frame #',
                      'nameLocation': 'middle',
                      'nameGap': 25,
                      'nameTextStyle': {
                          'fontSize': 14 
                      }},
            'yAxis': {
                'type': 'value',
                'name': 'Trunk Rotation (degrees)',
                'nameLocation': 'middle',  
                'nameRotate': 90,  # Rotates the title
                'nameTextStyle': {
                    'padding': [0, 0, 20, 0], 
                    'fontSize': 14,  
                }},
            'series': [{
                'type': 'line',
                'data': list(self.azimuthal_angle_data),
                'markLine': {  
                    'animation': False,
                    'data': [
                        {'xAxis': 0, 'lineStyle': {'color': 'black', 'width': 1}}
                    ]
                }
            }]
        })

        self.echart.style('width: 800px; height: 400px;')
        return self.echart
    
    def update_plot(self, frame_idx: int):
        self.echart.options['series'][0]['markLine']['data'][0]['xAxis'] = frame_idx
        self.echart.update()
        