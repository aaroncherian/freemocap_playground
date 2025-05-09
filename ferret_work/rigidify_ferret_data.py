from skellymodels.experimental.model_redo.tracker_info.model_info import ModelInfo
from skellymodels.experimental.model_redo.managers.human import Human
from pathlib import Path
import numpy as np

path_to_ferret_yaml = Path(__file__).parents[0]/'dlc_ferret.yaml'
ferret_model_info = ModelInfo(config_path=path_to_ferret_yaml)

path_to_3d_data = Path(r"C:\Users\aaron\Downloads\raw_dlc_3d_array_iteration_14_high_threshold.npy")

ferret = Human.from_landmarks_numpy_array(name="ferret",
               model_info=ferret_model_info,
               landmarks_numpy_array=np.load(path_to_3d_data))

ferret.calculate()
ferret.save_out_numpy_data(path_to_output_folder=Path(r"D:\ferret_recording\output_data"))



f = 2