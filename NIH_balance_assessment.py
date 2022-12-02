from freemocap_utils.freemocap_data_loader import FreeMoCapDataLoader
from path_length_tools import PathLengthCalculator

from pathlib import Path

path_to_freemocap_folder = Path(r'D:\ValidationStudy_aaron\FreeMoCap_Data')
sessionID = 'sesh_2022-11-02_13_55_55_atc_nih_balance'

loaded_freemocap_data = FreeMoCapDataLoader(path_to_freemocap_folder,sessionID)

freemocap_total_COM_data = loaded_freemocap_data.load_total_body_COM_data()
test_num_frame_range = range(1000,5000)

path_length_calculator = PathLengthCalculator.PathLengthCalculator(freemocap_total_COM_data)

#path_length_calculator.calculate_path_length(freemocap_body_data,test_num_frame_range)
path_length = path_length_calculator.get_path_length(test_num_frame_range)



f =2
