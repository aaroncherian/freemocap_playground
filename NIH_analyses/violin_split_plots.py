from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use("Qt5Agg")


path_to_qualisys_analysis_folder = Path(r'D:\ValidationStudy2022\FreeMocap_Data\qualisys_sesh_2022-05-24_16_02_53_JSM_T1_NIH\data_analysis\analysis_2023-03-14_11_23_07')
path_to_freemocap_analysis_folder = Path(r'D:\ValidationStudy2022\FreeMocap_Data\sesh_2022-05-24_16_02_53_JSM_T1_NIH\data_analysis\analysis_2023-03-13_12_58_36_2Hz')

# velocity_csv = f'condition_velocities_x.csv'
# path_to_qual_csv = path_to_qualisys_analysis_folder/velocity_csv

# file = open(path_to_qual_csv,'r')
# qual_data = list(csv.reader(file,delimiter = ','))
# file.close()

f = 2 

sns.set_theme(style = 'whitegrid')

for dimension in ['x','y','z']:
    velocity_csv = f'condition_velocities_{dimension}.csv'
    path_to_qual_csv = path_to_qualisys_analysis_folder/velocity_csv
    df_qual = pd.read_csv(path_to_qual_csv, index_col = False)
    df_qual.drop(columns=df_qual.columns[0], axis=1,  inplace=True)
    df_qual['System'] = 'qualisys'

    path_to_freemocap_csv = path_to_freemocap_analysis_folder/velocity_csv
    df_freemocap = pd.read_csv(path_to_freemocap_csv, index_col = False)
    df_freemocap.drop(columns=df_freemocap.columns[0], axis=1,  inplace=True)
    df_freemocap['System'] = 'freemocap'

    df_merged = pd.concat([df_freemocap,df_qual], ignore_index=False, sort = False)

    df_melted = pd.melt(df_merged, id_vars = 'System', value_vars = ['Eyes Open/Solid Ground', 'Eyes Closed/Solid Ground', 'Eyes Open/Foam', 'Eyes Closed/Foam'], var_name = 'Condition', value_name = 'COM_Velocity')
    ax = sns.violinplot(data = df_melted, x = 'Condition', y = 'COM_Velocity', hue = 'System', split = True, inner = 'quart',palette={"freemocap": "b", "qualisys": ".85"} )

    sns.despine(left=True)
    
    ax.set_xlabel('Condition')
    ax.set_ylabel(f'COM {dimension} Velocity')
    ax.set_title(f'COM {dimension} Velocity vs. Condition')
    ax.set_ylim([-1,1])
    fig = ax.get_figure()


    plt.show()


    fig.savefig(path_to_freemocap_analysis_folder/f'combined_violin_plot_{dimension}.png')
    f = 2