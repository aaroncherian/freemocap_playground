from pathlib import Path
from typing import Union

from anipose_utils import freemocap_anipose

def load_anipose_calibration_toml_from_path(
        camera_calibration_data_toml_path: Union[str, Path],
):
    try:
        anipose_calibration_object = freemocap_anipose.CameraGroup.load(str(camera_calibration_data_toml_path))

        return anipose_calibration_object
    except Exception as e:
        print(f"Failed to load anipose calibration info from {str(camera_calibration_data_toml_path)}")
        raise e

