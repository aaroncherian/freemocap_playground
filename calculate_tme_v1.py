import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Constants
GRAVITY = 9.81  # Gravitational constant in m/s^2
LBS_TO_KG_CONVERSION = 2.205  # Conversion factor from pounds to kilograms

def load_com_data(path: Path) -> np.ndarray:
    """
    Loads the center of mass data from the given file path.
    
    :param path: Path to the center of mass data file (npy format)
    :return: A NumPy array of the center of mass data
    """
    return np.load(path)

def convert_weight_to_mass(weight_lbs: float) -> float:
    """
    Converts weight in pounds to mass in kilograms.
    
    :param weight_lbs: Weight in pounds
    :return: Mass in kilograms
    """
    return weight_lbs / LBS_TO_KG_CONVERSION

def calculate_relative_height(z_positions: np.ndarray) -> np.ndarray:
    """
    Calculates the height relative to the minimum Z position (lowest point).
    
    :param z_positions: Array of Z-axis positions (height data)
    :return: Array of relative heights
    """
    min_height = np.min(z_positions)
    return z_positions - min_height  # Vectorized operation

def calculate_potential_energy(mass: float, relative_heights: np.ndarray) -> np.ndarray:
    """
    Calculates potential energy for each frame based on relative height.
    
    :param mass: Mass of the object (in kg)
    :param relative_heights: Relative heights (in meters)
    :return: Array of potential energy values for each frame
    """
    return mass * GRAVITY * relative_heights

def calculate_velocity(position_data: np.ndarray) -> np.ndarray:
    """
    Calculates the velocity based on position differences between frames.
    
    :param position_data: Array of 3D position data (Nx3 array)
    :return: Array of total velocities (magnitudes) between frames
    """
    velocity_data = np.diff(position_data, axis=0)  # Frame-to-frame differences
    return np.linalg.norm(velocity_data, axis=1)  # Magnitude of velocity vector

def calculate_kinetic_energy(mass: float, velocities: np.ndarray) -> np.ndarray:
    """
    Calculates the kinetic energy for each frame based on velocity.
    
    :param mass: Mass of the object (in kg)
    :param velocities: Array of velocities for each frame
    :return: Array of kinetic energy values for each frame
    """
    return 0.5 * mass * velocities**2

def ensure_array_lengths_match(array1: np.ndarray, array2: np.ndarray):
    """
    Trims the longer array to match the length of the shorter array.
    
    :param array1: First array
    :param array2: Second array
    :return: Tuple of trimmed arrays with matching lengths
    """
    min_length = min(len(array1), len(array2))
    return array1[:min_length], array2[:min_length]

def plot_mechanical_energy(com_z_data: np.ndarray, potential_energy: np.ndarray, 
                           kinetic_energy: np.ndarray, total_energy: np.ndarray) -> None:
    """
    Plots the Center of Mass Z trajectory, Potential Energy, Kinetic Energy, and Total Mechanical Energy.
    
    :param com_z_data: Z-axis data of the center of mass
    :param potential_energy: Potential energy data
    :param kinetic_energy: Kinetic energy data
    :param total_energy: Total mechanical energy data
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        subplot_titles=("Center of Mass Z Trajectory", "Potential Energy", 
                                        "Kinetic Energy", "Total Mechanical Energy"))

    fig.add_trace(go.Scatter(y=com_z_data, name='COM Z axis'), row=1, col=1,)
    fig.add_trace(go.Scatter(y=potential_energy, name='Potential Energy', line=dict(color='red')), row=2, col=1)
    fig.add_trace(go.Scatter(y=kinetic_energy, name='Kinetic Energy', line = dict(color = 'royalblue')), row=2, col=1)
    fig.add_trace(go.Scatter(y=total_energy, name='Total Mechanical Energy', line = dict(color = 'green', dash = 'dash')), row=2, col=1)

    fig.update_layout(height=1000, width=1600, title_text="Mechanical Energy Analysis")
    fig.show()

# Main process
def main(path_to_center_of_mass_data:Path, frames_to_use=None, weight_lbs=180):

    com_data = load_com_data(path_to_com_data)
    

    if frames_to_use:
        start_frame, end_frame = frames_to_use
        com_data = com_data[start_frame:end_frame]
    

    mass = convert_weight_to_mass(weight_lbs)

    relative_heights = calculate_relative_height(com_data[:, 2])
    potential_energy = calculate_potential_energy(mass, relative_heights)

    # Kinetic Energy Calculation
    velocities = calculate_velocity(com_data)
    kinetic_energy = calculate_kinetic_energy(mass, velocities)

    # Align lengths of potential and kinetic energy arrays
    potential_energy, kinetic_energy = ensure_array_lengths_match(potential_energy, kinetic_energy)

    # Total Mechanical Energy
    total_mechanical_energy = potential_energy + kinetic_energy

    plot_mechanical_energy(com_data[:, 2], potential_energy, kinetic_energy, total_mechanical_energy)
    # fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
    #                 subplot_titles=("Stacked Potential and Kinetic Energy", "Total Mechanical Energy with Deviations"))



# Execute the main process
if __name__ == "__main__":
    path_to_com_data = Path(r'D:\2024-08-01_treadmill_KK_JSM_ATC\1.0_recordings\sesh_2024-08-01_16_18_26_JSM_wrecking_ball\saved_data\npy\center_of_mass_frame_name_xyz.npy')
    weight_lbs = 180  # weight in pounds
    frames_to_use = (150,450)
    main(path_to_center_of_mass_data=path_to_com_data, frames_to_use=frames_to_use, weight_lbs=weight_lbs)
