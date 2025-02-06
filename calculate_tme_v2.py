import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

def calculate_time(data: np.ndarray, fps: float) -> np.ndarray:
    """
    Calculates the time array based on the number of frames and the frame rate.
    
    :param data: NumPy array of position data (Nx3 array)
    :param fps: Frame rate (frames per second)
    :return: Time array corresponding to each frame
    """
    return np.arange(len(data)) / fps

def calculate_velocity(position_data: np.ndarray, time: np.ndarray) -> np.ndarray:
    """
    Calculates 3D velocity using the Pythagorean theorem and converts from m/frame to m/sec.
    
    :param position_data: NumPy array of 3D position data (Nx3 array)
    :param time: NumPy array of time points
    :return: Velocity array in m/sec
    """
    dx_dy_dz = np.diff(position_data, axis=0)
    dt = np.diff(time)

    velocity = np.linalg.norm(dx_dy_dz, axis=1)/ dt
    
    # Add 0 at the beginning for alignment
    velocity = np.concatenate(([0], velocity))
    
    return velocity

def calculate_relative_height(z_positions: np.ndarray) -> np.ndarray:
    """
    Calculates the height relative to the minimum Z position (lowest point).
    
    :param z_positions: Array of Z-axis positions (height data)
    :return: Array of relative heights
    """
    min_height = np.min(z_positions)
    return z_positions - min_height  # Vectorized operation

def plot_all_energy(potential_energy: np.ndarray, kinetic_energy: np.ndarray, total_energy: np.ndarray, time: np.ndarray):
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, 
                        subplot_titles=("Potential, Kinetic and Total Mechanical Energy Over Time"))

    fig.add_trace(go.Scatter(x=time, y=potential_energy, name='Potential Energy', line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=time, y=kinetic_energy, name='Kinetic Energy', line=dict(color='royalblue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=time, y=total_energy, name='Total Mechanical Energy', line=dict(color='black', dash='dash')), row=1, col=1)

    fig.update_layout(
        height=800, 
        width=2000, 
        title_text="Mechanical Energy Analysis",
        xaxis_title="Time (s)",
        yaxis_title="Energy (J)"
    )
    fig.show()

def plot_all_energy_with_com_z(data:np.ndarray, potential_energy: np.ndarray, kinetic_energy: np.ndarray, total_energy: np.ndarray, time: np.ndarray):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        subplot_titles=("Potential, Kinetic and Total Mechanical Energy Over Time"))
    fig.add_trace(go.Scatter(x=time, y=data[:, 2], name='COM Z axis'), row=1, col=1)
    fig.add_trace(go.Scatter(x=time, y=potential_energy, name='Potential Energy', line=dict(color='red')), row=2, col=1)
    fig.add_trace(go.Scatter(x=time, y=kinetic_energy, name='Kinetic Energy', line=dict(color='royalblue')), row=2, col=1)
    fig.add_trace(go.Scatter(x=time, y=total_energy, name='Total Mechanical Energy', line=dict(color='black', dash='dash')), row=2, col=1)

    fig.update_layout(
        height=800, 
        width=2000, 
        title_text="Mechanical Energy Analysis",
        xaxis_title="Time (s)",
        yaxis_title="Energy (J)"
    )
    fig.show()

def plot_3d_trajectory(data, TME):
    # Get the range of the data for aspect ratio
    max_range = np.max([data[:, 0].max() - data[:, 0].min(),
                        data[:, 1].max() - data[:, 1].min(),
                        data[:, 2].max() - data[:, 2].min()])
    
    # Get the minimum and maximum of TME for color bar
    min_tme = np.min(TME)
    max_tme = np.max(TME)
    
    # Create the 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=data[:, 0], y=data[:, 1], z=data[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=TME,
            colorscale='Plasma',
            opacity=0.8,
            colorbar=dict(
                title="Kinetic Energy (J)",
                tickvals=[min_tme, max_tme],
                ticktext=[f"{min_tme:.2f} J", f"{max_tme:.2f} J"] 
            )
        )
    )])
    
    # Update layout for the 3D plot
    fig.update_layout(
        title='3D Trajectory Colored by Kinetic Energy',
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='cube',
            aspectratio=dict(x=1, y=1, z=1),
            xaxis=dict(range=[data[:, 0].mean() - max_range / 2, data[:, 0].mean() + max_range / 2]),
            yaxis=dict(range=[data[:, 1].mean() - max_range / 2, data[:, 1].mean() + max_range / 2]),
            zaxis=dict(range=[data[:, 2].mean() - max_range / 2, data[:, 2].mean() + max_range / 2])
        )
    )
    
    # Display the figure
    fig.show()

def plot_stacked_energy(time, PE, KE):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=PE, name='Potential Energy',
                             stackgroup='one', fillcolor='red'))
    fig.add_trace(go.Scatter(x=time, y=KE, name='Kinetic Energy',
                             stackgroup='one', fillcolor='blue'))
    fig.update_layout(
        title='Stacked Energy Over Time',
        yaxis_title='Energy (J)',
        xaxis_title='Time (s)'
    )
    fig.show()


def plot_tme_histogram(TME):
    fig = go.Figure(data=[go.Histogram(x=TME, nbinsx=50)])
    fig.update_layout(
        title='Distribution of Total Mechanical Energy',
        xaxis_title='Total Mechanical Energy (J)',
        yaxis_title='Frequency'
    )
    fig.show()


def plot_3d_trajectory_z_enlarged(data, TME):
    # Get the range of the data for x and y axes
    xy_max_range = np.max([data[:, 0].max() - data[:, 0].min(),
                           data[:, 1].max() - data[:, 1].min()])
    
    # Calculate z-axis range (but with a zoom-in on Z-axis)
    z_range = data[:, 2].max() - data[:, 2].min()
    z_center = (data[:, 2].max() + data[:, 2].min()) / 2
    z_margin = 0.05 * z_range  # Add 5% margin on the z-axis for better visibility

    # Get the minimum and maximum of TME for color bar
    min_tme = np.min(TME)
    max_tme = np.max(TME)
    
    # Create the 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=data[:, 0], y=data[:, 1], z=data[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=TME,
            colorscale='Plasma',
            opacity=0.8,
            colorbar=dict(
                title="Potential Energy (J)",
                tickvals=[min_tme, max_tme],
                ticktext=[f"{min_tme:.2f} J", f"{max_tme:.2f} J"] 
            )
        )
    )])
    
    # Update layout for the 3D plot
    fig.update_layout(
        title='3D Trajectory Colored by Potential Energy',
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.5),  # You can adjust the z squeezing factor here for a more focused look
            xaxis=dict(range=[data[:, 0].mean() - xy_max_range / 2, data[:, 0].mean() + xy_max_range / 2]),
            yaxis=dict(range=[data[:, 1].mean() - xy_max_range / 2, data[:, 1].mean() + xy_max_range / 2]),
            zaxis=dict(range=[data[:, 2].min() - z_margin, data[:, 2].max() + z_margin])  # Limit z-axis range to zoom in
        )
    )
    
    # Display the figure
    fig.show()


def plot_3d_trajectory_tme_deviation(data, TME):
    mean_TME = np.mean(TME)
    TME_deviation = TME - mean_TME
    max_range = np.max([data[:,0].max()-data[:,0].min(), 
                        data[:,1].max()-data[:,1].min(), 
                        data[:,2].max()-data[:,2].min()])
    
    fig = go.Figure(data=[go.Scatter3d(
        x=data[:,0], y=data[:,1], z=data[:,2],
        mode='markers',
        marker=dict(
            size=5,
            color=TME_deviation,
            colorscale='RdBu',
            colorbar=dict(title="TME Deviation (J)"),
            cmin=-np.max(np.abs(TME_deviation)),  # Symmetrical color scale
            cmax=np.max(np.abs(TME_deviation)),
            opacity=0.8
        )
    )])
    
    fig.update_layout(
        title='3D Trajectory Colored by TME Deviation from Mean',
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='cube',
            aspectratio=dict(x=1, y=1, z=1),
            xaxis=dict(range=[data[:,0].mean()-max_range/2, data[:,0].mean()+max_range/2]),
            yaxis=dict(range=[data[:,1].mean()-max_range/2, data[:,1].mean()+max_range/2]),
            zaxis=dict(range=[data[:,2].mean()-max_range/2, data[:,2].mean()+max_range/2])
        )
    )
    fig.show()


# Load data
path_to_com_data = Path(r"D:\2024-08-01_treadmill_KK_JSM_ATC\1.0_recordings\sesh_2024-08-01_16_22_49_JSM_kettlebell\saved_data\npy\center_of_mass_frame_name_xyz.npy")
weight_lbs = 180 #lbs this was just a random guess don't get mad Jon
fps = 30
frames_to_use = None # [start_frame, end_frame], specify None to use all frames or [start_frame, None] to use from start_frame to the end
# frames_to_use = [140, 500] # [start_frame, end_frame], specify None to use all frames or [start_frame, None] to use from start_frame to the end

data = np.load(path_to_com_data)

if frames_to_use:
    start_frame, end_frame = frames_to_use
    if end_frame is None:
        end_frame = data.shape[0]
    data = data[start_frame:end_frame]

time = calculate_time(data, 30) 
velocity = calculate_velocity(data, time)
relative_z = calculate_relative_height(data[:, 2])

# Constants
mass = weight_lbs/2.205  # kg
g = 9.81  # m/s^2

# Calculate energies
PE = mass * g * relative_z
KE = 0.5 * mass * velocity**2
TME = PE + KE

plot_all_energy(PE, KE, TME, time)
plot_3d_trajectory_z_enlarged(data, PE)
# plot_tme_histogram(TME)
# plot_3d_trajectory_tme_deviation(data, TME)
# plot_all_energy_with_com_z(data, PE, KE, TME, time)

# plot_phase_space(data, velocity)
# plot_stacked_energy(time, PE, KE)
# plot_velocity_profile(time, velocity)




# # Print some statistics
# print(f"Total data points: {len(data)}")
# print(f"Time range: {time[0]:.2f}s to {time[-1]:.2f}s")
# print(f"Average PE: {np.mean(PE):.2f} J")
# print(f"Average KE: {np.mean(KE):.2f} J")
# print(f"Average TME: {np.mean(TME):.2f} J")