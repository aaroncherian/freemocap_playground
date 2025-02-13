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
        with ui.matplotlib(figsize=(8, 4)).figure as fig:
            ax = fig.gca()
            # Plot the angle data
            ax.plot(self.frames, self.polar_angle_data, label='Trunk Inclination')
            
            # Add initial vertical line at frame 0
            self.vertical_line = ax.axvline(x=0, color='black', linestyle='-')
            
            # Customize plot
            ax.set_title('Trunk Inclination Over Time')
            ax.set_xlabel('Frame Number')
            ax.set_ylabel('Inclination Angle (Degrees)')
            ax.grid(True)
            ax.legend()
            
        self.plot = fig  # Store the figure for updates
        return fig
    
    def update_plot(self, frame_idx: int):
        # Update vertical line position
        self.vertical_line.set_xdata([frame_idx, frame_idx])
        # Need to redraw the figure
        self.plot.canvas.draw()