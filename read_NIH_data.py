
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#path_to_analysis_folder = Path(r'D:\ValidationStudy_aaron\FreeMoCap_Data\sesh_2022-11-02_13_55_55_atc_nih_balance\data_analysis\analysis_2023-01-24_12_39_57')
path_to_analysis_folder = Path(r'D:\ValidationStudy2022\FreeMocap_Data\sesh_2022-05-24_16_02_53_JSM_T1_NIH\data_analysis\analysis_2023-01-26_12_02_08')

velocity_csv = 'condition_velocities.csv'
path_to_csv = path_to_analysis_folder/velocity_csv

df = pd.read_csv(path_to_csv, index_col = False)

df.drop(columns=df.columns[0], axis=1,  inplace=True)

ax = sns.violinplot(data = df)

ax.set_xlabel('Condition')
ax.set_ylabel('COM Velocity')

fig = ax.get_figure()


plt.show()

fig.savefig(path_to_analysis_folder/'violin_plot.png')



f = 2

