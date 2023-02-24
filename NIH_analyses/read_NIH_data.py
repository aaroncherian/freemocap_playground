
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.collections import PathCollection


#path_to_analysis_folder = Path(r'D:\ValidationStudy_aaron\FreeMoCap_Data\sesh_2022-11-02_13_55_55_atc_nih_balance\data_analysis\analysis_2023-01-24_12_39_57')
#path_to_analysis_folder = Path(r'D:\ValidationStudy_aaron\FreeMoCap_Data\sesh_2022-11-02_13_55_55_atc_nih_balance\data_analysis\analysis_2023-01-27_16_38_20')
path_to_analysis_folder = Path(r'D:\ValidationStudy2022\FreeMocap_Data\qualisys_sesh_2022-05-24_16_02_53_JSM_T1_NIH\data_analysis\analysis_2023-02-21_14_56_38')
#path_to_analysis_folder = Path(r'D:\ValidationStudy2022\FreeMocap_Data\sesh_2022-05-24_16_02_53_JSM_T1_NIH\data_analysis\analysis_2023-01-30_13_12_51')

from freemocap_utils.GUI_widgets.NIH_widgets.path_length_tools import PathLengthCalculator


import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

for dimension in ['x','y','z']:
    velocity_csv = f'condition_velocities_{dimension}.csv'
    path_to_csv = path_to_analysis_folder/velocity_csv

    df = pd.read_csv(path_to_csv, index_col = False)

    df.drop(columns=df.columns[0], axis=1,  inplace=True)

    ax = sns.violinplot(data = df, cut = 1)

    # for artist in ax.lines:
    #     artist.set_zorder(10)
    # for artist in ax.findobj(PathCollection):
    #     artist.set_zorder(11)

    # #ax = sns.stripplot(data = df, jitter = False, zorder=1, alpha = .3)
    # ax = sns.swarmplot(data = df, zorder=1, size = .5)
 
    ax.set_xlabel('Condition')
    ax.set_ylabel(f'COM {dimension} Velocity')
    ax.set_title(f'COM {dimension} Velocity vs. Condition')

    fig = ax.get_figure()


    plt.show()

    # fig.savefig(path_to_analysis_folder/f'violin_swarm_plot_{dimension}.png')



tf = 2

