from models.skeleton import Skeleton
import numpy as np


def integrate_freemocap_data_into_skeleton_model(skeleton: Skeleton, freemocap_data: np.ndarray) -> Skeleton:
    """
    Integrates FreeMoCap 3D data into the skeleton model and calculates virtual markers if defined.

    Parameters:
    - skeleton: The Skeleton instance to integrate data into.
    - freemocap_data: A numpy array of 3D data points to be integrated.

    Returns:
    - The updated Skeleton instance with integrated 3D data and virtual markers calculated.
    """
    # Integrate 3D data into the skeleton model
    skeleton.integrate_freemocap_3d_data(freemocap_data)

    # Calculate virtual markers if they are defined in the marker hub
    if skeleton.markers.virtual_markers is not None:
        skeleton.calculate_virtual_markers()

    return skeleton