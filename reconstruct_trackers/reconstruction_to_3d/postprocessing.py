import numpy as np
import logging
from typing import List

# Configure logger
logger = logging.getLogger(__name__)


class ResultHandler:
    """Simple class to store and retrieve results from worker threads."""
    def __init__(self):
        self.result = None
    
    def handle_thread_finished(self, results, task_key):
        """Store the result from the specified task."""
        self.result = results[task_key]['result']


def process_and_filter_data(
    data_3d: np.ndarray, 
    landmark_names: List[str],
    cutoff_frequency: float = 6.0,
    sampling_rate: float = 90.0,
    filter_order: int = 4
) -> np.ndarray:
    """
    Process and filter 3D data using SkellyForge's filtering capabilities.
    
    Args:
        data_3d: 3D data to filter with shape (n_frames, n_points, 3)
        landmark_names: Names of the tracked landmarks
        cutoff_frequency: Cutoff frequency for the Butterworth filter
        sampling_rate: Sampling rate of the data in Hz
        filter_order: Order of the Butterworth filter
        
    Returns:
        Filtered 3D data with the same shape as input
    """
    from skellyforge.freemocap_utils.postprocessing_widgets.task_worker_thread import TaskWorkerThread
    from skellyforge.freemocap_utils.config import default_settings
    from skellyforge.freemocap_utils.constants import (
        TASK_FILTERING,
        PARAM_CUTOFF_FREQUENCY,
        PARAM_SAMPLING_RATE,
        PARAM_ORDER,
        PARAM_ROTATE_DATA,
        TASK_SKELETON_ROTATION,
        TASK_INTERPOLATION,
    )
    
    # Configure filter settings
    adjusted_settings = default_settings.copy()
    adjusted_settings[TASK_FILTERING][PARAM_CUTOFF_FREQUENCY] = cutoff_frequency
    adjusted_settings[TASK_FILTERING][PARAM_SAMPLING_RATE] = sampling_rate
    adjusted_settings[TASK_FILTERING][PARAM_ORDER] = filter_order
    adjusted_settings[TASK_SKELETON_ROTATION][PARAM_ROTATE_DATA] = False
    
    # Define tasks to perform
    task_list = [TASK_INTERPOLATION, TASK_FILTERING]
    
    # Set up result handler
    result_handler = ResultHandler()
    
    # Create and run worker thread
    worker_thread = TaskWorkerThread(
        raw_skeleton_data=data_3d,
        task_list=task_list,
        landmark_names=landmark_names,
        settings=adjusted_settings,
        all_tasks_finished_callback=lambda results: result_handler.handle_thread_finished(results, TASK_FILTERING),
    )

    worker_thread.start()
    worker_thread.join()
    
    return result_handler.result