import numpy as np
from pathlib import Path
from typing import Union 

def save_skeleton_array_to_npy(
    array_to_save: np.ndarray, skeleton_file_name: str, path_to_folder_where_we_will_save_this_data: Union[str, Path]
):
    if not skeleton_file_name.endswith(".npy"):
        skeleton_file_name += ".npy"
    Path(path_to_folder_where_we_will_save_this_data).mkdir(parents=True, exist_ok=True)
    np.save(
        str(Path(path_to_folder_where_we_will_save_this_data) / skeleton_file_name),
        array_to_save,
    )

