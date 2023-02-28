
from freemocap_utils import freemocap_data_loader
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

def calculate_magnitude(point):
    magnitude = np.sqrt((point[0])**2 + (point[1])**2 + (point[2])**2)
    return magnitude

#session_path = Path(r'D:\ValidationStudy_aaron\FreeMoCap_Data\sesh_2022-11-02_13_55_55_atc_nih_balance')

freemocap_session_path = Path(r'D:\ValidationStudy2022\FreeMocap_Data\sesh_2022-05-24_16_02_53_JSM_T1_NIH')
qualisys_session_path = Path(r'D:\ValidationStudy2022\FreeMocap_Data\qualisys_sesh_2022-05-24_16_02_53_JSM_T1_NIH')

path_list = [freemocap_session_path, qualisys_session_path]

condition_dataframe_dict = {}
sns.set_theme(style = 'whitegrid')
for system,path in zip(['freemocap', 'qualisys'],path_list):
    data_holder = freemocap_data_loader.FreeMoCapDataLoader(path)
    COM_data = data_holder.load_total_body_COM_data()
    velocity_COM = np.diff(COM_data, axis = 0)
    speed_list = []
    for x in range(velocity_COM.shape[0]):
        speed = calculate_magnitude(velocity_COM[x])
        speed_list.append(speed)

    eo_sg_range = [675,2325]
    ec_sg_range = [2725,4425]

    eo_fg_range = [6000,7700]
    ec_fg_range = [8350, 9850]

    eo_sg_data = speed_list[eo_sg_range[0]: eo_sg_range[1]]
    ec_sg_data = speed_list[ec_sg_range[0]: ec_sg_range[1]]
    eo_fg_data = speed_list[eo_fg_range[0]: eo_fg_range[1]]
    ec_fg_data = speed_list[ec_fg_range[0]: ec_fg_range[1]]
    
    condition_dict = {'EO/SG':eo_sg_data, 'EC/SG':ec_sg_data, 'EO/FG':eo_fg_data, 'EC/FG':ec_fg_data}

    condition_df = pd.DataFrame({ key:pd.Series(value) for key, value in condition_dict.items()})

    condition_df['System'] = system

    condition_dataframe_dict[system] = condition_df

df_merged = pd.concat([condition_dataframe_dict['freemocap'],condition_dataframe_dict['qualisys']], ignore_index=False, sort = False)

df_melted = pd.melt(df_merged, id_vars = 'System', value_vars = ['EO/SG', 'EC/SG', 'EO/FG', 'EC/FG'], var_name = 'Condition', value_name = 'COM_Speed')
ax = sns.violinplot(data = df_melted, x = 'Condition', y = 'COM_Speed', hue = 'System', split = True, cut = 0, inner = 'quart', palette={"freemocap": "b", "qualisys": ".85"} )

sns.despine(left=True)

ax.set_xlabel('Condition')
ax.set_ylabel(f'COM Speed')
ax.set_title(f'COM Speed vs. Condition')
fig = ax.get_figure()

plt.show()
f = 2
fig.savefig(freemocap_session_path/f'combined_violin_plot_speed.png')



# COM_data = data_holder.load_total_body_COM_data()
# velocity_COM = np.diff(COM_data, axis = 0)
# speed_list = []
# for x in range(velocity_COM.shape[0]):
#     speed = calculate_magnitude(velocity_COM[x])
#     speed_list.append(speed)


# eo_sg_range = [600,2170]
# ec_sg_range = [2950,4450]

# eo_fg_range = [5050,6550]
# ec_fg_range = [7000, 8500]


# ##for jon's data
# # eo_sg_range = [700,2350]
# # ec_sg_range = [2750,4400]

# # eo_fg_range = [6000,7750]
# # ec_fg_range = [8500, 9850]

# eo_sg_data = speed_list[eo_sg_range[0]: eo_sg_range[1]]
# ec_sg_data = speed_list[ec_sg_range[0]: ec_sg_range[1]]
# eo_fg_data = speed_list[eo_fg_range[0]: eo_fg_range[1]]
# ec_fg_data = speed_list[ec_fg_range[0]: ec_fg_range[1]]

# condition_dict = {'EO/SG':eo_sg_data, 'EC/SG':ec_sg_data, 'EO/FG':eo_fg_data, 'EC/FG':ec_fg_data}





# condition_df = pd.DataFrame({ key:pd.Series(value) for key, value in condition_dict.items()})


# ax = sns.violinplot(data = condition_df)

# ax = sns.violinplot(data = condition_df, cut = 1, color = ".8")

# for artist in ax.lines:
#     artist.set_zorder(10)
# for artist in ax.findobj(PathCollection):
#     artist.set_zorder(11)

# ax = sns.stripplot(data = condition_df, jitter = True, zorder=1, alpha = .3)

# ax.set_xlabel('Condition')
# ax.set_ylabel('COM Speed')

# ax.set_title('Speed Plot')

# fig = ax.get_figure()


# plt.show()


# f = 2


