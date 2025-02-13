
import numpy as np
import plotly.graph_objects as go

class BackAnglePlot:
    """3D visualization of spine angles"""
    def __init__(self, azimuthal, polar, vector_magnitude):
        # Calculate cartesian coordinates once during initialization
        self.x, self.y, self.z = self._calculate_cartesian(azimuthal, polar, vector_magnitude)
        # Set ranges once during initialization
        self._set_ranges()

    def _calculate_cartesian(self, azimuthal, polar, vector_magnitude):
        x = vector_magnitude * np.sin(polar) * np.cos(azimuthal)
        y = vector_magnitude * np.sin(polar) * np.sin(azimuthal)
        z = vector_magnitude * np.cos(polar)
        return x, y, z
    
    def _set_ranges(self):
        """Set the axis ranges based on data"""
        self.x_range = [float(min(min(self.x), 0)), float(max(max(self.x), 0))]
        self.y_range = [float(min(min(self.y), 0)), float(max(max(self.y), 0))]
        self.z_range = [float(min(min(self.z), 0)), float(max(max(self.z), 0))]
    
    def create_figure(self):
        """Create the initial plot figure"""
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=[0, float(self.x[0])],
            y=[0, float(self.y[0])], 
            z=[0, float(self.z[0])], 
            mode='lines+markers',
            name='Spine Vector',
            marker=dict(size=5),
            line=dict(width=3)
        ))

        fig.update_layout(
            title='Spine Vector Visualization',
            scene=dict(
                xaxis=dict(range=self.x_range),
                yaxis=dict(range=self.y_range),
                zaxis=dict(range=self.z_range),
                aspectmode='cube'
            ),
            showlegend=True,
            uirevision='true'  
        )
        return fig
    
    def update_plot(self, frame_idx):
        """Return the update data for a given frame"""
        return {
            'data': [{
                'type': 'scatter3d',
                'x': [0, float(self.x[frame_idx])],  
                'y': [0, float(self.y[frame_idx])],  
                'z': [0, float(self.z[frame_idx])],  
                'mode': 'lines+markers',
                'name': 'Spine Vector',
                'marker': {'size': 5},
                'line': {'width': 3}
            }],
            'layout': {
                'scene': {
                    'xaxis': {'range': self.x_range},
                    'yaxis': {'range': self.y_range},
                    'zaxis': {'range': self.z_range},
                    'aspectmode': 'cube'
                },
                'uirevision': 'true'
            }
        }