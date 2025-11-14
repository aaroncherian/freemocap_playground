from pathlib import Path
from skellymodels.managers.human import Human
import json

path_to_v2_recording = Path(r"C:\Users\aaron\Downloads\__freemocap_test_data\__freemocap_test_data")
path_to_save_data = Path(r"C:\Users\aaron\Documents\GitHub\nih_balance_analyses\docs\samples\test_data")


freemocap_human:Human = Human.from_data(path_to_v2_recording/'output_data')

freemocap_payload = {
    "positions": freemocap_human.body.xyz.as_array.tolist(),
    "connections": freemocap_human.body.anatomical_structure.segment_connections,
    "landmarks": freemocap_human.body.anatomical_structure.landmark_names}

# qualisys_human:Human = Human.from_data(path_to_recording/'validation'/'qualisys')
# qualisys_payload = {
#     "positions": qualisys_human.body.xyz.as_array.tolist(),
#     "connections": qualisys_human.body.anatomical_structure.segment_connections,
#     "landmarks": qualisys_human.body.anatomical_structure.landmark_names
# }


Path(path_to_save_data/"freemocap_data.json").write_text(json.dumps(freemocap_payload))
# Path(path_to_save_data/"qualisys_data.json").write_text(json.dumps(qualisys_payload))

f = 2