import numpy as np
import logging
from typing import List

# Configure logger
logger = logging.getLogger(__name__)


from skellyforge.freemocap_utils.postprocessing_widgets.postprocessing_functions import interpolate_data, filter_data

def postprocess_data(
        data_3d: np.ndarray,
        cutoff_frequency: float = 6.0,
        sampling_rate: float = 30.0,
        filter_order: int = 4,
):
    # Interpolate missing data
    data_3d = interpolate_data.interpolate_skeleton_data(data_3d)

    # Apply filtering
    data_3d = filter_data.filter_skeleton_data(
        data_3d,
        cutoff=cutoff_frequency,
        sampling_rate=sampling_rate,
        order=filter_order
    )

    return data_3d
