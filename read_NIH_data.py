
from pathlib import Path
import pandas as pd

path_to_analysis_folder = Path(r'D:\ValidationStudy_aaron\FreeMoCap_Data\sesh_2022-11-02_13_55_55_atc_nih_balance\data_analysis\analysis_2023-01-24_12_39_57')

velocity_csv = 'condition_velocities.csv'
path_to_csv = path_to_analysis_folder/velocity_csv

df = pd.read_csv(path_to_csv, index_col = False)

df.drop(columns=df.columns[0], axis=1,  inplace=True)

velocities_array = df.T.to_numpy()


f = 2

